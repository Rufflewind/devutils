import functools, itertools, logging, os, pickle
import re, subprocess, tempfile, traceback, types
from . import utils

logger = logging.getLogger(__name__)

def _format_stack(stack):
    return "".join(traceback.format_list(stack))

class Symbol(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class Meta(object):
    '''An object provides equality over the first value, ignoring the second.'''
    def __init__(self, value, meta):
        self.value = value
        self.meta = meta

    def __repr__(self):
        return "Meta{!r}".format((self.value, self.meta))

    def __eq__(self, other):
        return self.value == other.value

class UnionConflictError(ValueError):
    def __init__(self, key, value1, value2, *args):
        self.key = key
        self.value1 = value1
        self.value2 = value2
        message = ("conflicting values for {!r}: {!r} != {!r}"
                   .format(key, value1, value2))
        super(UnionConflictError, self).__init__(message, *args)

class UnionDict(object):
    '''A wrapper over a dictionary where |= is overloaded to perform merges.
    The merge fails if there are keys with differing values.'''

    def __init__(self, inner=()):
        self.inner = dict(inner)

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
                    raise UnionConflictError(k, self_v, v)
                continue
            self.inner[k] = v
        return self

class Make(object):
    '''A `Make` object represents either:

      - a rule: a single rule along with a collection of its dependencies; or
      - a ruleset: an anonymous collection of rules (dependencies).

    A rule contains a name, dependencies, commands, and attributes.

      - A name is a string.  It is the target of the makefile rule.
      - Dependencies are an ordered sequence of `Make` objects.  A dependency
        is included in the prerequisites of the makefile rule if and only if
        it is a rule, not a ruleset.
      - Commands are an ordered sequences of strings.  In the makefile, each
        string is translated to a single line preceded by a tab.
      - Attributes are lists of pairs of the form (key, value).  The key is a
        unique identifier for this particular attribute type.  The value is an
        arbitrary object that forms a semilattice using the `|=` operator.

    Ruleset contain only dependencies and attributes, just like rules.  Their
    names are always `None` and their commands must be an empty sequence.

    A useful trick is to anonymize a rule by wrapping it: `Make(None, rule)`.
    This allows them to be attached as a dependency without being included
    as an explicit prerequisite.  This is done automatically by
    `InferenceRule.make()`, for example.

    A `Make` object may be created with a lazily-populated sequence of
    dependencies.  This means passing a function rather than a list to the
    `deps` argument of the constructor.  When `populate_deps(cache)` is
    called, the function will be called with the `cache` argument, where
    `cache` should be a `dict`-like object.  Until `populate_deps` is called,
    the `.deps` attribute will raise an exception.
    '''

    @staticmethod
    def ensure(make):
        if isinstance(make, str):
            return Make(make)
        elif isinstance(make, Make):
            return make
        else:
            raise TypeError("an object of {!r} cannot be converted to Make"
                            .format(type(make)))

    @staticmethod
    def _check_args(name, deps, cmds, attrs):
        if name is None and cmds:
            raise TypeError("if name is None, cmds must be empty: {!r}"
                            .format(cmds))
        if not callable(deps):
            deps = [Make.ensure(dep) for dep in deps]
        if callable(cmds):
            if callable(deps):
                raise TypeError("deps and cmds cannot both be callables")
            cmds = cmds(*(dep.name for dep in deps))
        if not (isinstance(cmds, tuple) or isinstance(cmds, list)):
            raise TypeError("cmds must be either list or tuple, "
                            "or a callable that returns such: {!r}"
                            .format(cmds))
        for cmd in cmds:
            if not isinstance(cmd, str):
                raise TypeError("cmds must be a list of str: {!r}".format(cmds))
        try:
            valid_attrs = all(callable(attr) or
                              (isinstance(attr, tuple) and
                               len(attr) == 2)
                              for attr in attrs)
        except TypeError:
            valid_attrs = False
        if not valid_attrs:
            raise TypeError("attrs must be a list of callables and/or "
                            "pairs: {!r}".format(attrs))
        return deps, cmds

    def __init__(self, name=None, deps=(), cmds=(), attrs=()):
        '''The `deps` argument can be either `deps` or a function `(cache) ->
        deps`.  In the latter case the `.deps` field will remain unavailable
        until `.populate_deps()` is called.

        The string dependencies are automatically promoted to `Make` objects
        using `Make.ensure`.

        Instead of a list of strings, `cmds` can be a function `(*deps) ->
        [str]`.  In this case, `deps` must not be a function.

        `attrs` can contain be either `(key, value)` pairs, or functions of
        the form `(name) -> (attr_key, value)`.'''
        deps, cmds = self._check_args(name, deps, cmds, attrs)
        self.name = name
        self.raw_deps = deps
        self.cmds = cmds
        self.attrs = {}
        self.update_attrs(*attrs)
        self._stack = traceback.extract_stack()

    def __repr__(self):
        deps = ("<unpopulated>" if callable(self.raw_deps) else
                "[{}]".format(", ".join("<Make for {!r} at {}>"
                                        .format(dep.name, hex(id(dep)))
                                        for dep in self.deps)))
        tup = (self.name, Symbol(deps), self.cmds) + tuple(
            ((Symbol(k.__name__) if callable(k) else k), v)
            for k, v in self.attrs.items())
        return "Make{!r}".format(tup)

    def __ior__(self, other):
        self.deps.append(other)
        return self

    def with_attrs(self, *attrs):
        '''Make a copy with the given attributes amended.'''
        copy = Make(self.name, self.raw_deps, self.cmds)
        copy.attrs = dict(self.attrs)
        copy._stack = self._stack
        copy.update_attrs(*attrs)
        return copy

    def update_attrs(self, *attrs):
        '''Amend the given attributes.'''
        attrs = list(attrs)
        for i, attr in enumerate(attrs):
            if not attr:
                continue
            if callable(attr):
                if self.name is None:
                    raise TypeError("function attributes are not supported "
                                    "if name is None")
                attrs[i] = attr(self.name)
        for k, v in attrs:
            try:
                old_v = self.attrs[k]
            except KeyError:
                self.attrs[k] = v
            else:
                old_v |= v
                self.attrs[k] = old_v

    @property
    def is_trivial(self):
        '''
        Return whether the `Make` object needs to be rendered.

        Raises an error if the list of dependencies has not yet been populated.
        '''
        return self.name is None or (
            all(dep.name is None for dep in self.deps) and
            not (self.cmds or self.attrs))

    @property
    def deps(self):
        '''Obtain the dependencies of the rule.  This will raise an error if
        the dependencies have not yet been populated.'''
        raw_deps = self.raw_deps
        if callable(raw_deps):
            raise ValueError("dependencies haven't been populated yet")
        return raw_deps

    def populate_deps(self, cache):
        '''Populate the dependencies of the current node.

        This has no effect if the dependencies are already populated.'''
        raw_deps = self.raw_deps
        if callable(raw_deps):
            deps = raw_deps(cache)
            # avoid subtle bugs caused by dependency functions that return
            # stateful objects like generators (confusingly, they would be
            # reported as conflicts that with identical stack traces!)
            if not (isinstance(deps, list) or isinstance(deps, tuple)):
                raise TypeError("the dependency function must return either "
                                "a tuple or a list: {!r}".format(deps))
            self.raw_deps = [Make.ensure(dep) for dep in deps]

    def append(self, dep):
        '''
        Append `dep` to the list of dependencies and return `dep`.

        Raises an error if the list of dependencies has not yet been populated.
        '''
        self.deps.append(dep)
        return dep

    def walk(self):
        '''Traverse first through self and then through all its unique
        transitive dependencies in depth-first order.  Send a true-ish
        value to skip the dependencies of a particular node.

        Requires a tree with fully populated dependencies.'''
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

    def merged_attr(self, attr_key, accum):
        '''Attributes are expected to form a semilattice with respect to the
        |= operator.

        Requires a tree with fully populated dependencies.'''
        for m in self.walk():
            try:
                attr = m.attrs[attr_key]
            except KeyError:
                continue
            accum |= attr
        return accum

    def default_target(self):
        '''
        Find the name of the first rule within the transitive closure.
        Returns `None` if there are no rules.

        Requires a tree with fully populated dependencies.'''
        for m in self.walk():
            t = self.name
            if t is not None:
                return t
        return None

    def gather_macros(self):
        '''Requires a tree with fully populated dependencies.'''
        try:
            macros = self.merged_attr(MACROS, UnionDict()).inner
            err = None
        except UnionConflictError as e:
            err = e
        if err:
            raise ValueError("conflicting attributes for {!r}:\n  "
                             "{!r}\n  {!r}\n\n"
                             "The former attribute comes from:\n{}\n"
                             "The latter attribute comes from:\n{}"
                             .format(err.key,
                                     err.value1.value,
                                     err.value2.value,
                                     _format_stack(err.value1.meta),
                                     _format_stack(err.value2.meta))
                             .rstrip())
        return dict((k, v.value) for k, v in macros.items())

    def gather_suffixes(self):
        '''Requires a tree with fully populated dependencies.'''
        suffixes = sorted(self.merged_attr(SUFFIXES, set()))
        return Make(".SUFFIXES", suffixes) if suffixes else MakeSet()

    def gather_phony(self):
        '''Requires a tree with fully populated dependencies.'''
        phonys = sorted(self.merged_attr(PHONY, set()))
        return Make(".PHONY", phonys) if phonys else MakeSet()

    def gather_clean(self, exclusions):
        '''Requires a tree with fully populated dependencies.'''
        clean_cmds = self.merged_attr(CLEAN_CMDS, set())
        cleans = self.merged_attr(CLEAN, set())
        cleans.difference_update(exclusions)
        if cleans:
            clean_cmds.add("rm -fr " + " ".join(sorted(cleans)))
        return (Make("clean", (), sorted(clean_cmds))
                if clean_cmds else MakeSet())

    def render_rule(self):
        if self.name is None:
            raise TypeError("cannot render a rule whose name is None: {!r}"
                            .format(self))
        return "".join((
            self.name,
            ":",
            "".join(" " + dep.name
                    for dep in self.deps
                    if dep.name is not None),
            "\n",
            "".join("\t{}\n".format(cmd) for cmd in self.cmds),
        ))

    def render(self, f, out_name="Makefile", cache=None):
        # make sure modifications are localized
        self = self.with_attrs()
        self.raw_deps = list(self.deps)

        cache = {} if cache is None else cache
        for make in self.walk():
            make.populate_deps(cache)

        default_target = self.default_target()
        macros = self.gather_macros()
        suffixes = self.gather_suffixes()
        phony = self.gather_phony()
        phony_names = set(dep.name for dep in phony.deps)
        assert None not in phony_names
        clean = self.gather_clean(phony_names)
        extras = (suffixes, phony, clean)

        seen = {}
        # categorize rules so we can re-order them later
        first_rule = []
        phony_rules = []
        normal_rules = []
        self_rule = []
        inference_rules = []
        special_rules = []
        special_phony_rule = []
        for make in itertools.chain(self.walk(), extras):
            # skip trivial rules to avoid unnecessary bloat and conflicts
            if make.is_trivial:
                continue

            name = make.name
            rendered = make.render_rule()
            old_rule = seen.get(name)

            # check for conflicting rules
            if old_rule is not None:
                if old_rule[0] != rendered:
                    old_make = old_rule[1]
                    raise ValueError("conflicting rules:\n  {!r}\n  {!r}\n\n"
                                     "The former rule comes from:\n{}\n"
                                     "The latter rule comes from:\n{}"
                                     .format(old_make, make,
                                             _format_stack(old_make._stack),
                                             _format_stack(make._stack))
                                     .rstrip())
                continue

            seen[name] = (rendered, make)
            if name == default_target:
                first_rule.append(rendered)
            elif name == out_name:
                self_rule.append(rendered)
            elif re.match("\.[^.]+(\.[^.]+)", name) and not make.deps:
                inference_rules.append(rendered)
            elif name == ".PHONY":
                special_phony_rule.append(rendered)
            elif re.match("\.[A-Z]", name) or name == out_name:
                special_rules.append(rendered)
            elif name in phony_names:
                phony_rules.append(rendered)
            else:
                normal_rules.append(rendered)
        phony_rules.sort()
        inference_rules.sort()
        special_rules.sort()
        normal_rules.sort()
        rules = itertools.chain(first_rule,
                                phony_rules,
                                normal_rules,
                                self_rule,
                                inference_rules,
                                special_rules,
                                special_phony_rule)

        for k, v in macros.items():
            f.write("{}={}\n".format(k, v))
        for i, rule in enumerate(rules):
            if i != 0 or macros:
                f.write("\n")
            f.write(rule)

    def save(self, path, out_name=None, cache=None):
        cache_path = None
        if cache is not None and isinstance(cache, str):
            cache_path = cache
            try:
                cache = pickle.loads(utils.load_file(cache_path, binary=True))
            except OSError:
                cache = {}

        with utils.safe_open(path, "w") as f:
            self.render(f, out_name=out_name or path, cache=cache)
            f.flush()

        if cache_path is not None:
            try:
                utils.save_file(cache_path, pickle.dumps(cache), binary=True)
            except OSError:
                logger.warn("could not save cache to file: {}"
                            .format(cache_path))

def make(name=None, deps=(), cmds=(), attrs=(),
         clean=True, mkdir=True, phony=False, macros={}):
    '''A slightly more high-level wrapper over the `Make` constructor.

    Unlike the bare `Make` constructor, this one by default adds a clean rule
    (`clean=True`) and prepends the command with `mkdir` (`mkdir=True`).
    It also supports the `phony` and `macros` arguments for convenience.

    Note that if the rule is phony or the name is `None`, `clean` and `mkdir`
    are ignored, which means this would have the same result as the bare
    `Make` constructor.
    '''
    if phony:
        attrs = (PHONY,) + tuple(attrs)
    if macros:
        attrs = (MACROS(macros),) + tuple(attrs)
    m = Make(name, deps, cmds, attrs)
    if name is not None and CLEAN not in m.attrs and not m.attrs.get(PHONY):
        if clean:
            m.update_attrs(CLEAN)
        if mkdir:
            m.cmds = ("mkdir -p $(@D)",) + tuple(m.cmds)
    return m

def MACROS(macros=()):
    try:
        macros = dict(macros)
    except TypeError as e:
        raise TypeError("'macros' should be a dict-like object")
    stack = traceback.extract_stack()
    return MACROS, dict((k, Meta(v, stack)) for k, v in macros.items())

def SUFFIXES(suffixes):
    return SUFFIXES, set(suffixes)

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
            if isinstance(result, types.GeneratorType):
                raise TypeError("cannot cache generators")
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
    key = (run_dep_tool, dep_tool, path)
    try:
        old_mtime, result = cache[key]
    except (IndexError, KeyError):
        old_mtime = None
    mtime = os.stat(path).st_mtime
    if old_mtime is None or mtime > old_mtime:
        logger.debug("running dep tool for: {}".format(path))
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
            raise RuntimeError("failed to run dependency tool for {}:\n\n{1}"
                               .format(repr(path), err))
        # warn about missing dependencies because these dependencies
        # themselves may also contain dependencies that we don't know; to fix
        # this, it is recommended to run `make` to generate the missing files,
        # and then run `makegen` again.
        dep_paths = []
        for dep_path in make_rule_header_parse(out):
            if not os.path.exists(dep_path):
                logger.warning("missing dependency: {}".format(dep_path))
            dep_paths.append(dep_path)
        # note: we are caching the result, so don't use a generator!
        result = tuple(dep_paths)
        cache[key] = mtime, result
    else:
        logger.debug("already cached; skipping dep tool for: {}".format(path))
    return result

def c_dep_tool(in_fn, out_fn):
    # -MG prevents the compiler from complaining about missing headers
    return ("cc", "-MG", "-MM", "-MF", out_fn, in_fn)

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
        cmds = (("mkdir -p $(@D)",) if mkdir else ()) + tuple(self.cmds)
        return Make(deps=(Make(self.src_ext + self.tar_ext,
                               cmds=cmds,
                               attrs=(SUFFIXES(suffixes),)),))

C_INFERENCE_RULE = InferenceRule(".c", ".o",
                                 ("$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<",))

C_MPI_INFERENCE_RULE = InferenceRule(
    ".c", ".o_mpi",
    ("$(MPICC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<",))

def make_c_obj(path, via=C_INFERENCE_RULE):
    ''''via' can either be an InferenceRule or (name, cmds)'''
    if isinstance(path, Make):
        return path
    base_deps_func = functools.partial(C_DEPS_FUNC, path)
    if isinstance(via, InferenceRule):
        stem, _ = os.path.splitext(path)
        name = stem + via.tar_ext
        cmds = ()
        deps_func = lambda cache: base_deps_func(cache) + (via.make(),)
    else:
        name, cmds = via
        deps_func = base_deps_func
    return make(name, deps_func, cmds, mkdir=False)

def make_bin(name,
             deps,
             ld="$(CC) $(CFLAGS)",
             libs="$(LIBS)",
             make_obj=make_c_obj):
    if isinstance(deps, str):
        raise TypeError("'deps' argument must be a list, not str")
    deps = tuple(tuple(dep if isinstance(dep, Make) else make_obj(dep)
                       for dep in deps))
    dep_names = " ".join(dep.name for dep in deps if dep.name is not None)
    cmds = ("{ld} -o $@ {dep_names} {libs}".format(**locals()),)
    return make(name, deps, cmds)

def make_vpath(subdir, targets=["all"], macros={}, vpath_patterns=None):
    '''Create a sub-build using the same Makefile but within another directory
    using VPATH/vpath.  This can be useful for running the same build under a
    different environment without polluting the main tree.

    If vpath_patterns is provided, vpath is used; otherwise, VPATH is used.
    Prefer vpath over VPATH to prevent output files of non-VPATH builds from
    contaminating the sub-build.

    GNU Make is required to use this feature.'''
    vpath_patterns = sorted(vpath_patterns)
    vpath = os.path.join(os.path.relpath(".", subdir), "")
    if vpath_patterns is not None:
        vpaths = ["@echo vpath {} {} >>$@/Makefile".format(pattern, vpath)
                  for pattern in vpath_patterns]
    else:
        vpaths = ["@echo VPATH={} >>$@/Makefile".format(vpath)]
    targets = " ".join(sorted(targets))
    macros = " ".join("{}='{}'".format(k, v) for k, v in sorted(macros.items()))
    return make(subdir,
                [],
                ["@printf '' >$@/Makefile"] + vpaths + [
                    "@echo _all: {} >>$@/Makefile".format(targets),
                    "@echo include {}Makefile >>$@/Makefile".format(vpath),
                    "$(MAKE) -C $@ {}".format(macros),
                ],
                [PHONY])
