import functools, itertools, logging, os, re, subprocess, tempfile, traceback

logger = logging.getLogger(__name__)

class MakeError(Exception):
    pass

class Symbol(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class UnionDict(object):
    def __init__(self, d=()):
        self.inner = dict(d)

    def __repr__(self):
        return "UnionDict({!r})".format(self.inner)

    def __ior__(self, other):
        for k, v in other.items():
            try:
                self_v = self.inner[k]
            except KeyError:
                pass
            else:
                if self_v != v:
                    raise ValueError("conflicting values for {!r}: {!r} != {!r}"
                                     .format(k, self_v, v))
                continue
            self.inner[k] = v
        return self

class Identity(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Identity({!r})".format(self.value)

    def __hash__(self):
        __hash__ = getattr(self.value, "__hash__", None)
        if __hash__ is not None:
            return __hash__()
        return hash(id(self.value))

    def __eq__(self, other):
        __hash__ = getattr(self.value, "__hash__", None)
        if __hash__ is not None:
            return self.value == other
        return id(self.value) == id(other.value)

class Make(object):

    @staticmethod
    def ensure(make):
        if isinstance(make, str):
            return Make(make)
        elif isinstance(make, Make):
            return make
        else:
            raise TypeError("an object of {!r} cannot be converted to Make"
                            .format(type(make)))

    def __init__(self, name, raw_deps=(), cmds=(), *attrs):
        '''The 'raw_deps' argument can be either [deps] or (cache) -> deps.
        In the latter case the '.deps' field will remain unavailable until
        '.populate_deps()' is called.

        'attrs' can contain either (attr_key, value), (name) -> (attr_key,
        value), or (attr_key, None).  The last one prevents attributes of
        dependencies from propagating upward.'''
        if name is None and cmds:
            raise TypeError("if name is None, cmds must be None too: {!r}"
                            .format(cmds))
        if not (isinstance(cmds, tuple) or isinstance(cmds, list)):
            raise TypeError("cmds must either list or tuple: {!r}"
                            .format(cmds))
        for cmd in cmds:
            if not isinstance(cmd, str):
                raise TypeError("cmds must be a list of str: {!r}"
                                .format(cmds))
        if not callable(raw_deps):
            raw_deps = [Make.ensure(dep) for dep in raw_deps]
        self.name = name
        self.raw_deps = raw_deps
        self.cmds = cmds
        self.attrs = {}
        self.update_attrs(*attrs)
        self._stack = traceback.extract_stack()[:-1]

    def __repr__(self):
        deps = "<unpopulated>" if callable(self.raw_deps) else "<populated>"
        tup = (self.name, Symbol(deps), self.cmds) + tuple(
            ((Symbol(k.__name__) if callable(k) else k), v)
            for k, v in self.attrs.items())
        return "Make{!r}".format(tup)

    def with_attrs(self, *attrs):
        copy = Make(self.name, self.raw_deps, self.cmds)
        copy.attrs = dict(self.attrs)
        copy.update_attrs(*attrs)
        return copy

    def update_attrs(self, *attrs):
        attrs = list(attrs)
        for i, attr in enumerate(attrs):
            if not attr:
                continue
            if callable(attr):
                if self.name is None:
                    raise TypeError("function attributes are not supported "
                                    "if name is None")
                attrs[i] = attr(self.name)
        self.attrs.update(attrs)

    @property
    def macros(self):
        try:
            return self.attrs[MACROS]
        except KeyError:
            pass
        macros = {}
        self.attrs[MACROS] = macros
        return macros

    @property
    def is_trivial(self):
        return not self.deps and not self.cmds and (
            not self.attrs or (len(self.attrs) == 1 and
                               AUXILIARY in self.attrs))

    @property
    def deps(self):
        raw_deps = self.raw_deps
        if callable(raw_deps):
            raise ValueError("dependencies haven't been populated yet")
        return raw_deps

    def populate_deps(self, cache):
        raw_deps = self.raw_deps
        if callable(raw_deps):
            deps = raw_deps(cache)
            self.raw_deps = [Make.ensure(dep) for dep in deps]

    def append(self, dep):
        self.deps.append(dep)
        return dep

    def traverse(self):
        '''Traverse first through self and then through all its unique
        transitive dependencies in an unspecified order.  Send a true-ish
        value to skip the dependencies of a particular node.'''
        seen = set()
        candidates = [self]
        while candidates:
            candidate = candidates.pop()
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            skip_deps = yield candidate
            if not skip_deps:
                candidates.extend(reversed(candidate.deps))
            candidates.extend(candidate.attrs.get(AUXILIARY, ()))

    def merged_attr(self, attr_key, accum):
        '''Attributes are expected to form a semilattice with respect to the
        |= operator.'''
        traversal = self.traverse()
        for m in traversal:
            while True:
                try:
                    attr = m.attrs[attr_key]
                except KeyError:
                    break
                if attr is None:
                    try:
                        m = traversal.send(True)
                    except StopIteration:
                        return
                    continue
                accum |= attr
                break
        return accum

    def default_target(self):
        t = self.name
        if t is not None:
            return t
        for dep in self.deps:
            t = dep.default_target()
            if t is not None:
                return t
        return None

    def gather_suffixes(self):
        suffixes = sorted(self.merged_attr(SUFFIXES, set()))
        return Make(".SUFFIXES", suffixes) if suffixes else MakeSet()

    def gather_phony(self):
        phonys = sorted(self.merged_attr(PHONY, set()))
        return Make(".PHONY", phonys) if phonys else MakeSet()

    def gather_clean(self, exclusions):
        clean_cmds = self.merged_attr(CLEAN_CMDS, set())
        cleans = self.merged_attr(CLEAN, set())
        cleans.difference_update(exclusions)
        if cleans:
            clean_cmds.add("rm -fr " + " ".join(cleans))
        return (Make("clean", (), sorted(clean_cmds))
                if clean_cmds else MakeSet())

    def render_rule(self):
        if self.name is None:
            raise TypeError("cannot render a rule whose name is None: {!r}"
                            .format(self))
        return "".join((
            self.name,
            ":",
            "".join(" " + dep.name for dep in self.deps),
            "\n",
            "".join("\t{}\n".format(cmd) for cmd in self.cmds),
        ))

    def render(self, f):
        # make sure modifications are localized
        self = self.with_attrs()
        self.raw_deps = list(self.deps)

        cache = {}
        for make in self.traverse():
            make.populate_deps(cache)

        macros = self.merged_attr(MACROS, UnionDict()).inner
        suffixes = self.gather_suffixes()
        phony = self.gather_phony()
        clean = self.gather_clean(dep.name for dep in phony.deps)
        extras = (suffixes, phony, clean)

        rules = {}
        default_target = self.default_target()
        for make in itertools.chain(itertools.islice(self.traverse(), 1, None),
                                    extras):
            # skip trivial rules to avoid unnecessary bloat + conflicts
            if make.is_trivial:
                continue
            name = make.name
            rule = (make.render_rule(), make)
            old_rule = rules.get(name)
            if old_rule is not None and old_rule[0] != rule[0]:
                old_stack = "".join(traceback.format_list(old_rule[1]._stack))
                stack = "".join(traceback.format_list(rule[1]._stack))
                raise MakeError("conflicting rules:\n  {!r}\n  {!r}\n\n"
                                "Origin of first rule:\n{}\n"
                                "Origin of second rule:\n{}"
                                .format(old_rule[1], rule[1], old_stack, stack))
            rules[name] = rule
        first_rule = (() if default_target is None
                      else (rules.pop(default_target),))

        for k, v in macros.items():
            f.write("{}={}\n".format(k, v))
        if macros:
            f.write("\n")

        for rule, _ in itertools.chain(first_rule, sorted(rules.values())):
            f.write(rule + "\n")

def make(name, deps=(), cmds=(), *attrs, clean=True, mkdir=True):
    m = Make(name, deps, cmds, *attrs)
    if CLEAN not in m.attrs and not m.attrs.get(PHONY):
        if clean:
            m.update_attrs(CLEAN)
        if mkdir:
            m.cmds = ("mkdir -p $(@D)",) + tuple(m.cmds)
    return m

def MACROS(macros=()):
    assert not isinstance(macros, str)
    return MACROS, macros

def SUFFIXES(suffixes):
    return SUFFIXES, set(suffixes)

def AUXILIARY(rules):
    if isinstance(rules, str):
        raise TypeError("expected an iterable of of Make")
    return AUXILIARY, set(rules)

def PHONY(name):
    return PHONY, set((name,))

def CLEAN(name):
    return CLEAN, set((name,))

def CLEAN_CMDS(cmds):
    assert not isinstance(cmds, str)
    return CLEAN_CMDS, set(cmds)

def memoize(cache, key):
    def inner(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return cache[key]
            except (IndexError, KeyError):
                pass
            result = f(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return inner

def make_rule_header_lex(string):
    '''This is a simplified lexer that is not faithful to the actual makefile
    syntax (colons and tabs are ignored) but it is sufficient for parsing the
    output of cpp -M.'''
    import re
    token = ""
    escaping = False
    for m in re.finditer(r"([^ \n\\]*)([ \n\\]?)", string):
        s, c = m.groups()
        token += s
        if escaping:
            escaping = False
            if not s:
                if c == "\n":
                    token = token[:-1]
                    if token:
                        yield token[:-1]
                        token = ""
                else:
                    token += c
                continue
        if c == "\\":
            token += "\\"
            escaping = True
        else:
            if token:
                yield token
                token = ""
            if c == "\n":
                break
    if token:
        yield token

def make_rule_header_parse(string):
    import re
    # ignore the target (btw, this regex is surprisingly robust)
    m = re.match("(?s).*?: (.*)", string)
    if not m:
        raise ValueError("could not parse dependency tool output:\n\n" + string)
    return make_rule_header_lex(m.group(1))

def run_dep_tool(dep_tool, path, cache=None):
    '''dep_tool(source_path: str, output_path: str) -> [str]'''
    @memoize(cache, (run_dep_tool, dep_tool, path))
    def get():
        with tempfile.NamedTemporaryFile(mode="r") as f:
            with open(os.devnull, "wb") as fnull:
                p = subprocess.Popen(dep_tool(path, f.name),
                                     stdin=fnull,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
            err, _ = p.communicate()
            out = f.read()
        if p.returncode or not out:
            raise MakeError("failed to run dependency tool for {0}:\n\n{1}"
                            .format(repr(path), err))
        # warn about missing dependencies because these dependencies
        # themselves may also contain dependencies that we don't know; to fix
        # this, it is recommended to run `make` to generate the missing files,
        # and then run `makegen` again.
        for dep_path in make_rule_header_parse(out):
            if not os.path.exists(dep_path):
                logger.warning("missing dependency: {0}".format(dep_path))
            yield dep_path
    return get()

def c_dep_tool(in_fn, out_fn):
    return ("cc", "-MM", "-MF", out_fn, in_fn)

C_DEPS_FUNC = functools.partial(run_dep_tool, c_dep_tool)

class InferenceRule(object):
    def __init__(self, src_ext, tar_ext, cmds):
        self.src_ext = src_ext
        self.tar_ext = tar_ext
        self.cmds = cmds

    def __repr__(self):
        return "InferenceRule{!r}".format(
            (self.src_ext, self.tar_ext, self.cmds))

    def make(self, mkdir=True):
        suffixes = [self.src_ext]
        if self.tar_ext:
            suffixes.append(self.tar_ext)
        return Make(self.src_ext + self.tar_ext,
                    [],
                    (("mkdir -p $(@D)",) if mkdir else ()) + tuple(self.cmds),
                    SUFFIXES(suffixes))

C_INFERENCE_RULE = InferenceRule(".c", ".o",
                                 ("$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<",))

def make_c_obj(path, via=C_INFERENCE_RULE):
    ''''via' can either be an InferenceRule or (name, cmds)'''
    if isinstance(path, Make):
        return path
    if not path.endswith(".c"):
        raise ValueError("expected source file (*.c), got: {!r}"
                         .format(path))
    if isinstance(via, InferenceRule):
        stem, _ = os.path.splitext(path)
        name = stem + via.tar_ext
        cmds = ()
        attrs = (AUXILIARY((via.make(),)),)
    else:
        name, cmds = via
        attrs = ()
    return make(name, functools.partial(C_DEPS_FUNC, path), cmds, *attrs,
                mkdir=False)

def make_bin(name,
             deps,
             ld="$(CC) $(CFLAGS)",
             libs="$(LIBS)",
             make_obj=make_c_obj):
    if isinstance(deps, str):
        raise TypeError("'deps' argument must be a list, not str")
    deps = tuple(tuple(dep if isinstance(dep, Make) else make_obj(dep)
                       for dep in deps))
    dep_names = " ".join(dep.name for dep in deps)
    return make(name,
                deps,
                ("{ld} -o $@ {dep_names} {libs}\n".format(**locals()),))
