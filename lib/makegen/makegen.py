# ----------------------------------------------------------------------------
# Generic utilities
# -----------------

#@snip/do_nothing[
def do_nothing(*args, **kwargs):
    pass
#@]

#@snip/update_dict[
def update_dict(d0, *dicts, **kwargs):
    merger = kwargs.pop("merger", None)
    for k in kwargs:
        raise TypeError("got an unexpected keyword argument {0}".format(k))
    for d in dicts:
        if merger:
            for k, v in d.items():
                exists = False
                try:
                    v0 = d0[k]
                    exists = True
                except KeyError:
                    pass
                if exists:
                    d0[k] = merger((k, v0), (k, v))[1]
                else:
                    d0[k] = v
        else:
            d0.update(d)
#@]

#@snip/merge_dicts[
def merge_dicts(*dicts, **kwargs):
    merger = kwargs.pop("merger", None)
    for k in kwargs:
        raise TypeError("got an unexpected keyword argument {0}".format(k))
    d0 = {}
    update_dict(d0, *dicts, merger=merger)
    return d0
#@]

#@snip/merge_sets[
def merge_sets(*sets):
    s0 = set()
    for s in sets:
        s0.update(s)
    return s0
#@]

def freeze_value(value):
    '''Convert a value into an immutable form with a total ordering.'''
    if isinstance(value, tuple) or isinstance(value, list):
        return tuple(map(freeze_value, value))
    elif isinstance(value, dict):
        return tuple(sorted((k, freeze_value(v)) for k, v in value.items()))
    elif isinstance(value, set) or isinstance(value, frozenset):
        return tuple(sorted(value))
    return value

def exclusive_merge(arg0, *args):
    '''Return one of the arguments if all of them are equal.  Fails with
    `ValueError` otherwise.'''
    for arg in args:
        if arg0 != arg:
            raise ValueError("conflicting values: {0} vs {1}"
                             .format(repr(arg0), repr(arg)))
    return arg0

def merge_frozen_dicts(*dicts):
    '''Merge several dictionaries with the values frozen.  Fails with
    `ValueError` if they have conflicting values for the same key.'''
    return merge_dicts(*(
        dict((k, freeze_value(v))
             for k, v in d.items())
        for d in dicts
    ), merger=exclusive_merge)

def toposort_sortnodes(graph, nodes, reverse=False):
    '''Sort nodes by the number of immediate dependencies, followed by the
    nodes themselves.'''
    return sorted(nodes, key=(lambda node: (len(graph[node]), node)),
                  reverse=reverse)

def toposort_countrdeps(graph):
    '''Count the number of immediate dependents (reverse dependencies).
    Returns a dict that maps nodes to number of dependents, as well as a list
    of roots (nodes with no dependents).'''
    numrdeps = {}
    for node, deps in graph.items():
        for dep in deps:
            numrdeps[dep] = numrdeps.get(dep, 0) + 1
    roots = []
    for node, deps in graph.items():
        if node not in numrdeps:
            numrdeps[node] = 0
            roots.append(node)
    return numrdeps, roots

def toposort(graph, reverse=False):
    '''Topologically sort a directed acyclic graph, ensuring that dependents
    are placed after their dependencies, or the reverse if `reverse` is true.

        graph: {node: [node, ...], ...}

    The `graph` is a dictionary of nodes: the key is an arbitrary value that
    uniquely identifies the node, while the value is an iterable of
    dependencies for that node.  For example:

        graph = {0: [1, 2], 1: [2], 2: []}

    This is a graph where 0 depends on both 1 and 2, and 1 depends on 2.

    The sorted result is always deterministic.  However, to achieve this,
    nodes are required to form a total ordering.'''

    # make sure there are no duplicate edges
    graph = dict((node, set(deps)) for node, deps in graph.items())

    # count the number of dependents and extract the roots
    numrdeps, roots = toposort_countrdeps(graph)

    # sort nodes to ensure a deterministic topo-sorted result; current algo
    # sorts by # of immediate dependencies followed by node ID, so nodes with
    # fewer immediate dependencies and/or lower node IDs tend to come first
    roots = toposort_sortnodes(graph, roots, reverse=reverse)
    graph = dict((node, toposort_sortnodes(graph, deps, reverse=reverse))
                 for node, deps in graph.items())

    # Kahn's algorithm
    # (note: this will alter numrdeps and roots)
    result = []
    while roots:
        node1 = roots.pop()
        result.append(node1)
        for node2 in graph[node1]:
            numrdeps[node2] -= 1
            if not numrdeps[node2]:
                roots.append(node2)
    if len(result) != len(graph):
        raise ValueError("graph is cyclic")
    if not reverse:
        result.reverse()
    return result

def simple_repr(value, name):
    return "{0}({1})".format(name, ", ".join(
        "{0}={1}".format(k, repr(v))
        for k, v in sorted(value.__dict__.items())
    ))

# ----------------------------------------------------------------------------
# Hashing
# -------

DEFAULT_HASH_LENGTH = 20

def hash_str(s, length=DEFAULT_HASH_LENGTH):
    import base64, hashlib
    h = hashlib.sha512(s.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h)[:length].decode("ascii")

def hash_json(data, length=DEFAULT_HASH_LENGTH):
    import json
    s = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hash_str(s, length=length)

# ----------------------------------------------------------------------------
# Syntactic manipulation
# ----------------------

def snormpath(path):
    import re
    sep = "/"
    m = re.match(re.escape(sep) + "*", path)
    num_leading_slashes = len(m.group(0))
    if num_leading_slashes > 2:
        num_leading_slashes = 1
    return (sep * num_leading_slashes +
            sep.join(s for s in path.split(sep)
                     if not re.match(r"\.?$", s))) or "."

def shell_quote(string):
    import re
    # we try to be conservative here because some shells have more special
    # characters than others (`!` and `^` are not safe); we require empty
    # strings to be quoted
    if re.match("[]a-zA-Z0-9/@:.,_+-]+$", string):
        return string
    return "'{0}'".format(string.replace("'", "'\\''"))

def shell_quote_arg(string):
    import re
    # allow `=` since it's safe as an argument
    if re.match("[]a-zA-Z0-9/@:.,_=+-]+$", string):
        return string
    return shell_quote(string)

def make_escape(string):
    return string.replace("$", "$$")

def make_unescape(string):
    return string.replace("$$", "$")

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

def save_makefile(file, macros_rules, header="", posix=False):
    '''
    macros_rules: ({Str: Str, ...},
                   [([Str, ...], [Str, ...], [Str, ...]), ...])
    header: Str
    posix: Bool

    `rules` is a list of triples each containing the targets, prerequisites,
    and commands respectively.
    '''
    macros, rules = macros_rules
    # do some sanity checks to avoid subtle bugs
    for value in macros.values():
        if not isinstance(value, str):
            raise ValueError("macro value must be a str: " +
                             repr(value))
    for targets, prerequisites, commands in rules:
        if isinstance(targets, str):
            raise ValueError("targets must be a list of str, "
                             "not a str: " + repr(targets))
        if isinstance(prerequisites, str):
            raise ValueError("prerequisites must be a list of str, "
                             "not a str: " + repr(prerequisites))
        if isinstance(commands, str):
            raise ValueError("commands must be a list of str, "
                             "not a str: " + repr(commands))
    with open(file, "wt") as stream:
        stream.write(header)
        in_block = False
        if posix:
            stream.write(".POSIX:\n")
            in_block = True

        if in_block:
            stream.write("\n")
            in_block = False

        for i, (name, value) in enumerate(sorted(macros.items())):
            in_block = True
            stream.write("{0}={1}\n".format(name, value))

        for targets, prerequisites, commands in rules:
            if in_block:
                stream.write("\n")
            in_block = True
            stream.write(" ".join(targets))
            stream.write(":")
            stream.write("".join(" " + x for x in sorted(prerequisites)))
            stream.write("\n")
            stream.write("".join("\t" + x + "\n" for x in commands))

# ----------------------------------------------------------------------------
# Source code dependencies
# ------------------------
#
# A dependency tool is a function that takes the filename of the source file
# and returns a list of filenames that the given source file depends on
# (usually due to a preprocessing mechanism such as `#include`).
#
#     DependencyTool = (Str) -> [Str, ...]

def guess_source_code_language(extension):
    if extension.startswith("."):
        extension = extension[1:]
    if extension == "c":
        return "c"
    if extension in ["cc", "cpp", "cxx", "c++"]:
        return "c++"
    raise ValueError("cannot guess language for ." + extension)

def get_linker_flags(objs, libraries):
    import itertools
    libs = libraries
    stdlibs = set()
    for obj in objs:
        libs = itertools.chain(libs, obj.hint["libraries"])
        stdlibs.update(obj.hint["standard_libraries"])

    # find the compiler that can cover the most num of standard libs
    if ".c++" in stdlibs:
        stdlibs.remove(".c++")
        stdlibs.remove(".c")
        language = "c++"
    elif ".f" in stdlibs:
        stdlibs.remove(".f")
        stdlibs.remove(".c")
        language = "f"
    else:
        stdlibs.remove(".c")
        language = "c"

    # gather all the libraries and toposort them
    superlib = Library("", libs)
    libs = toposort(superlib.graph, reverse=True)
    libs.remove("")
    macros = superlib.macros
    del superlib

    # fill in the gaps by explicitly linking the remaining standard libs
    if ".f" in stdlibs:
        stdlibs.remove(".f")
        update_dict(macros, {"LIBFORTRAN": "gfortran"},
                    merger=exclusive_merge)
        libs.append("$(LIBFORTRAN)")

    if stdlibs:
        raise ValueError("unknown standard libraries: " + " ".join(stdlibs))

    return language, libs, macros

def null_dependency_tool(filename):
    return []

def run_external_dependency_tool(command, filename):
    import os, subprocess, tempfile
    with open(os.devnull, "wb") as fnull, \
         tempfile.NamedTemporaryFile() as out_fn:
        p = subprocess.Popen(command + [out_fn.name, filename],
                             stdin=fnull,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)
        err, _ = p.communicate()
        with open(out_fn.name, "rt") as f:
            out = f.read()
    if p.returncode or not out:
        raise ValueError("failed to run dependency tool for {0}:\n\n{1}"
                         .format(repr(filename), err))
    return sorted(make_rule_header_parse(out))

def check_external_dependency_tool(command, extension):
    import os, subprocess, tempfile
    with open(os.devnull, "wb") as fnull, \
         tempfile.NamedTemporaryFile() as out_fn, \
         tempfile.NamedTemporaryFile(suffix=extension) as in_fn:
        return subprocess.Popen(
            command + [out_fn.name, in_fn.name],
            stdin=fnull,
            stdout=fnull,
            stderr=fnull,
        ).wait() == 0

def guess_default_dependency_tool(language):
    import os
    if language == "c":
        cc = os.environ.get("CC", "cc")
        ext = ".c"
        candidates = [[cc] + std + ["-MG", "-MM", "-MF"] for std in [
            ["-std=c11"],
            ["-std=c99"],
            ["-std=c89"],
            [],
        ]]
    elif language == "c++":
        cxx = os.environ.get("CXX", "c++")
        ext = ".cpp"
        candidates = [[cxx] + std + ["-MG", "-MM", "-MF"] for std in [
            ["-std=c++17"],
            ["-std=c++14"],
            ["-std=c++11"],
            ["-std=c++03"],
            [],
        ]]
    else:
        raise ValueError("unknown language: {0}".format(repr(language)))
    try:
        return next(cmd for cmd in candidates
                    if check_external_dependency_tool(cmd, ext))
    except StopIteration:
        raise Exception("no dependency tool available for {0}"
                        .format(language))

def get_dependency_tool(language):
    if language not in DEPENDENCY_TOOLS:
        return null_dependency_tool
    tool = DEPENDENCY_TOOLS[language]
    if not tool:
        import functools
        tool = functools.partial(
            run_external_dependency_tool,
            guess_default_dependency_tool(language),
        )
        DEPENDENCY_TOOLS[language] = tool
    return tool

'''A dict of dependency tools keyed by language.  This variable may be tweaked
as necessary.'''
DEPENDENCY_TOOLS = {
    "c": None,
    "c++": None,
}

# ----------------------------------------------------------------------------
# Building makefiles
# ------------------

def get_suffixes(inference_rules):
    '''[(Str, *), ...] -> {Str, ...}'''
    import re
    suffixes = set()
    for rule in inference_rules:
        suffixes.update(re.match(r"(\.[^.]+)(\.[^.]+)$", rule[0]).groups())
    return suffixes

def is_path_prefix(subpath, path):
    import os
    # a bit of a hack
    if not subpath:
        subpath = "."
    return ".." not in os.path.relpath(subpath, path).split("/")

def auto_mkdir(commands, out_fn=None, dep_fns=None):
    import re, os
    commands = list(commands)
    if out_fn is not None and not re.search(r"[$/\\]", out_fn):
        return commands
    out_dir = os.path.dirname(out_fn)
    if (dep_fns and
        not any("$" in f for f in dep_fns) and
        "$" not in out_fn and
        any(is_path_prefix(out_dir, os.path.dirname(f)) for f in dep_fns)):
        return commands
    d = "`dirname $@`"
    if "$" not in out_fn:
        d = os.path.dirname(out_fn)
    return ["@mkdir -p {0}".format(d)] + commands

def prettify_rules(rules):
    # regroup rules that are identical in prerequisites and commands
    merged_rules = merge_dicts(*(
        {freeze_value((sorted(prereqs), commands)): frozenset([target])}
        for target, (prereqs, commands) in rules.items()
    ), merger=lambda x, y: (x[0], merge_sets(x[1], y[1])))
    return sorted((tuple(sorted(targets)),) + prereqs_commands
                  for prereqs_commands, targets in merged_rules.items())

def prettify_inference_rules(inference_rules):
    return sorted(((tuple([target]), (), commands)
                   for target, commands in inference_rules.items()),
                  key=(lambda rule: rule[0]))

def emit_clean_rule(cleans, clean_cmds):
    if isinstance(clean_cmds, str):
        raise ValueError("clean_cmds should be a list-like, not str")
    return {"clean": (
        [],
        sorted((["rm -fr --" + "".join(
            " {0}".format(shell_quote_arg(x))
            for x in sorted(cleans)
        )] if cleans else []) + list(clean_cmds)),
    )}

def emit_makefile(macros, default_target, phonys, rules,
                       inference_rules, special_rules):
    macros = dict((k, v) for k, v in macros.items() if v is not None)
    rules = dict(rules)              # make a copy as we're about to modify it
    default_rule = ({} if default_target is None else
                    {default_target: rules.pop(default_target)})
    phony_rules = dict((phony, rules.pop(phony))
                       for phony in phonys if phony in rules)
    return (macros, (
        prettify_rules(default_rule) +
        prettify_rules(phony_rules) +
        prettify_rules(rules) +
        prettify_inference_rules(inference_rules) +
        special_rules
    ))

class Ruleset(object):
    '''The attributes should never be mutated, because they may be shared with
    various other rulesets.  Instead, replace them with a copy if you wish to
    edit them.  Better yet, just make a new ruleset.'''

    def __init__(self, rules={}, macros={}, inference_rules={},
                 cleans=None, clean_cmds=frozenset(),
                 phonys=frozenset(), default_target=None,
                 hint=None):
        '''
        rules: {Str: ([Str], [Str]), ...}
        macros: {Str: Str, ...}
        inference_rules: {Str: [Str], ...}
        cleans: {Str, ...}
        clean_cmds: {Str, ...}
        phonys: {Str, ...}
        default_target: Str | None
        hint: Any
        '''
        if "clean" in rules:
            raise ValueError("the 'clean' rule is reserved")
        for phony in phonys:
            if phony not in rules:
                raise ValueError("phony target does not exist: " + phony)
        if default_target is None:
            if phonys:
                default_target = min(phonys)
            elif rules:
                default_target = min(rules)
        if cleans is None:
            cleans = frozenset(rules.keys()).difference(phonys)
        self.rules = rules
        self.macros = macros
        self.inference_rules = inference_rules
        self.cleans = cleans
        self.clean_cmds = clean_cmds
        self.phonys = phonys
        self.default_target = default_target
        self.hint = hint

    def __repr__(self):
        return simple_repr(self, "Ruleset")

    def merge(self, *others, **kwargs):
        '''Merge several rulesets with the current one and return the result.
        The merge is slightly biased: it will use the first default target
        that is not 'None'.'''
        hint_merger = kwargs.pop("hint_merger", exclusive_merge)
        for k in kwargs:
            raise TypeError("got an unexpected keyword argument {0}".format(k))
        others = [self] + list(others)
        try:
            default_target = next(x.default_target for x in others
                                  if x.default_target is not None)
        except StopIteration:
            default_target = None
        return Ruleset(
            rules=merge_frozen_dicts(*(x.rules for x in others)),
            macros=merge_dicts(*(x.macros for x in others),
                               merger=exclusive_merge),
            inference_rules=merge_frozen_dicts(*(x.inference_rules
                                                 for x in others)),
            cleans=merge_sets(*(x.cleans for x in others)),
            clean_cmds=merge_sets(*(x.clean_cmds for x in others)),
            phonys=merge_sets(*(x.phonys for x in others)),
            default_target=default_target,
            hint=hint_merger(*(x.hint for x in others)),
        )

    def copy(self):
        return self.merge()

    def emit(self):
        clean_rule = emit_clean_rule(self.cleans, self.clean_cmds)
        phonys = self.phonys.union(clean_rule.keys())
        suffixes = get_suffixes(self.inference_rules.items())
        secondaries = ([([".SECONDARY"], sorted(self.cleans), [])]
                       if self.cleans and getattr(self, "secondary", False)
                       else [])
        special_rules = (
            ([([".SUFFIXES"], sorted(suffixes), [])] if suffixes else []) +
            ([([".PHONY"], sorted(phonys), [])] if phonys else []) +
            secondaries
        )
        return emit_makefile(
            macros=self.macros,
            default_target=self.default_target,
            phonys=phonys,
            rules=merge_frozen_dicts(self.rules, clean_rule),
            inference_rules=self.inference_rules,
            special_rules=special_rules,
        )

    def save(self, file="Makefile"):
        save_makefile(file, self.emit())

def alias(name, rulesets):
    rulesets = tuple(rulesets)
    prereqs = frozenset(r.default_target for r in rulesets)
    return Ruleset(
        rules={name: (prereqs, [])},
        phonys=[name],
    ).merge(*rulesets)

def emit_args(args):
    if isinstance(args, str):
        # fail early to avoid subtle bugs
        raise TypeError("must be a list of str, not a str: " + repr(args))
    return "".join(" " + shell_quote_arg(arg) for arg in args)

def cpp_macros_to_flags(macros):
    return ["-D{0}={1}".format(k, v) for k, v in sorted(macros.items())]

def get_clike_language_info(language):
    if language == "c":
        return "$(CC)", "$(CFLAGS)"
    elif language == "c++":
        return "$(CXX)", "$(CXXFLAGS)"
    else:
        raise ValueError("not a C-like language: {0}".format(repr(language)))

def emit_compile_args(language, args):
    import itertools
    compiler, flags = get_clike_language_info(language)
    extra_flags = args["extra_flags"]
    if not isinstance(extra_flags, str):
        extra_flags = emit_args(extra_flags)
    if extra_flags:
        extra_flags = " " + extra_flags
    return "".join([
        (
            compiler if args["compiler"] is None else
            emit_args(args["compiler"])
        ),
        "" if not args["inherit_cppflags"] else " $(CPPFLAGS)",
        "" if not args["inherit_flags"] else " " + flags,
        emit_args(itertools.chain(
            (
                [] if args["standard"] is None else
                ["-std={0}".format(args["standard"])]
            ),
            cpp_macros_to_flags(args["macros"]),
        )),
        extra_flags,
    ])

def compile_source(filename, language=None, out_filename=None,
                   out_directory=None, libraries=[], detect_dependencies=True,
                   directory=".", suffix=None, extra_deps=[], **kwargs):
    '''Compile a source file into an object file.  The file can be given as a
    `str` or as another `Ruleset` that generates the source file, in which
    case `detect_dependencies` is automatically turned off.'''
    import os, logging
    if isinstance(filename, Ruleset):
        dependency = filename
        filename = filename.default_target
        detect_dependencies = False
    stem, ext = os.path.splitext(filename)
    language = language or guess_source_code_language(ext)
    args = merge_dicts(DEFAULT_ARGS[language], kwargs)
    prereqs = [d.default_target for d in extra_deps]
    if detect_dependencies:
        if os.path.isfile(filename):
            try:
                prereqs.extend(get_dependency_tool(language)(filename))
            except Exception as e:
                logging.warn(e)
        else:
            logging.warn("cannot detect dependencies (file does not exist) "
                         "for: {0}".format(filename))
    prereqs.append(filename)
    compile_args = emit_compile_args(language, args)
    args_hash = hash_str(compile_args)
    simple_case = args_hash == DEFAULT_ARGS_HASH[language]
    simple_out_fn = snormpath(stem + ".o")
    if suffix is None:
        suffix = "" if simple_case else "_" + args_hash
    default_out_fn = snormpath("{0}{1}.o".format(stem, suffix))
    if out_filename is None:
        if out_directory is None:
            out_filename = default_out_fn
        else:
            out_filename = snormpath(os.path.join(
                out_directory,
                os.path.basename(default_out_fn),
            ))
    elif out_directory is not None:
        raise ValueError("out_filename and out_directory cannot be both given")
    out_filename = snormpath(out_filename)
    invisible_rule = False
    prereqs = frozenset(prereqs)
    if simple_case and out_filename == simple_out_fn:
        commands = []
        inference_rules = INFERENCE_RULES[language]
        if len(prereqs) == 1:
            invisible_rule = True
    else:
        commands = auto_mkdir([
            "{0} -c -o $@ {1}".format(
                compile_args,
                make_escape(shell_quote(filename)),
            ),
        ], out_fn=out_filename, dep_fns=prereqs)
        inference_rules = {}
    return Ruleset(
        rules={out_filename: (prereqs, commands)},
        inference_rules=inference_rules,
        macros=DEFAULT_MACROS[language],
        hint={
            "libraries": tuple(libraries),
            "standard_libraries": frozenset(STANDARD_LIBS[language]),
            "invisible_rule": invisible_rule,
        },
    )

def build_program(out_filename, objs, libraries=[], extra_flags="",
                  extra_deps=[]):
    import os
    objs = tuple(objs)
    obj_fns = [obj.default_target for obj in objs]
    if isinstance(libraries, str):
        customlibs = " " + libraries
        libraries = []
    else:
        customlibs = ""
    language, libs, macros = get_linker_flags(objs, libraries)
    compiler, flags = get_clike_language_info(language)
    if extra_flags:
        extra_flags += " "
    return Ruleset(
        rules={out_filename: (
            frozenset(obj_fns + [d.default_target for d in extra_deps]),
            auto_mkdir(["{0} {1} {5}-o $@ {2}{3}{4}".format(
                compiler,
                flags,
                " ".join(
                    make_escape(shell_quote(obj_fn))
                    for obj_fn in obj_fns
                ),
                "".join(" -l" + x for x in libs),
                customlibs,
                extra_flags,
            )], out_fn=out_filename),
        )},
        macros=merge_dicts(DEFAULT_MACROS[language], macros),
    ).merge(*(obj for obj in objs
              if not obj.hint.get("invisible_rule", False)),
            hint_merger=do_nothing)

class Library(object):

    def __init__(self, name, deps=[], macro=None):
        deps = tuple(deps)
        graph = {name: tuple(sorted(dep.name for dep in deps))}
        update_dict(graph, *(dep.graph for dep in deps),
                    merger=exclusive_merge)
        macros = {}
        if macro:
            name, value = macro
            macros[name] = value
        update_dict(macros, *(dep.macros for dep in deps),
                    merger=exclusive_merge)
        self.name = name
        self.graph = graph
        self.macros = macros

    def __repr__(self):
        return simple_repr(self, "Library")

INFERENCE_RULES = {
    "c": {".c.o": [
        "$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<",
    ]},
    "c++": {".cpp.o": [
        "$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<",
    ]},
}

DEFAULT_MACROS = {
    "c": {}, # omitted because it is part of POSIX
    "c++": {}, # omitted because most practical implementations have it
}

DEFAULT_CLIKE_ARGS = {
    "compiler": None, # or [Str, ...]
    "inherit_cppflags": True,
    "inherit_flags": True,
    "standard": None, # or Str
    "macros": {},
    "extra_flags": [],
}

DEFAULT_ARGS = {
    "c": DEFAULT_CLIKE_ARGS,
    "c++": DEFAULT_CLIKE_ARGS,
}

DEFAULT_ARGS_HASH = dict(
    (language, hash_str(emit_compile_args(language, args)))
    for language, args in DEFAULT_ARGS.items()
)

STANDARD_LIBS = {
    "c": [".c"],
    "c++": [".c", ".c++"],
}

def separate_dependencies(dependencies):
    fns = []
    rulesets = []
    for dep in dependencies:
        if isinstance(dep, str):
            fns.append(dep)
        elif isinstance(dep, Ruleset):
            fns.append(dep.default_target)
            rulesets.append(dep)
        else:
            raise TypeError("invalid dependency type: {0!r}".format(dep))
    return fns, rulesets

def plain_file(fn):
    return Ruleset(default_target=fn)

def simple_command(command, out_filename, dependencies=[],
                   no_clean=False, phony=False):
    import os
    dep_fns, deps = separate_dependencies(dependencies)
    kwargs = {
        "out": "'$@'",
        "all": " ".join(map(shell_quote, dep_fns)),
        "all1": " ".join(map(shell_quote, dep_fns[1:])),
    }
    if isinstance(command, str):
        command = command.split("\n")
    commands = [cmd.format(*dep_fns, **kwargs) for cmd in command]
    commands = auto_mkdir(commands, out_filename, dep_fns=dep_fns)
    return Ruleset(
        rules={out_filename: (frozenset(dep_fns), commands)},
        cleans=frozenset() if no_clean else None,
        phonys=[out_filename] if phony else [],
    ).merge(*deps)

def download(url, out_filename=None, out_directory=".",
             no_clean=True, checksum=None):
    import os
    try:
        import urllib.parse as urllib_parse
    except ImportError:
        import urlparse as urllib_parse
    if out_filename is None:
        name = os.path.basename(urllib_parse.urlparse(url).path)
        if not name:
            name = "index.html"
        out_filename = os.path.join(out_directory, name)
    out_filename = snormpath(out_filename)
    commands = auto_mkdir([
        " # downloading {0} to $@ ...".format(url),
        "@if command >/dev/null 2>&1 -v curl; then "
            "curl -fLSs -o $@.tmp {1}; "
        "elif command >/dev/null 2>&1 -v wget; then "
            "wget -nv -O $@.tmp {1}; "
        "else "
            'echo >&2 "error: can\'t download '
                'without curl or wget installed"; '
            "exit 1; "
        "fi".format(url, shell_quote_arg(url)),
    ], out_filename)
    if checksum:
        checksum_type, checksum_value = checksum
        commands.extend([
            "@printf '%s  %s' {0} $@.tmp >$@.tmp.{1}sum"
                .format(checksum_value, checksum_type),
            "@{0}sum -c --quiet -- $@.tmp.{0}sum"
                .format(checksum_type),
        ])
    commands.append("@mv $@.tmp $@")
    return Ruleset(
        rules={out_filename: (frozenset(), commands)},
        cleans=frozenset() if no_clean else None,
    ).merge(*dependencies)

def relocate(filename, prefix=".", root="."):
    import os
    return snormpath(os.path.join(root, prefix,
                                  os.path.relpath(filename, root)))

class RelocatedBuilder(object):

    def __init__(self, root=".", bindir="dist/bin", tmpdir="dist/tmp"):
        '''Note: bindir and tmpdir are relative to root, not current dir.'''
        self.root = root
        self.bindir = bindir
        self.tmpdir = tmpdir

    def compile_source(self, filename, *args, **kwargs):
        import os
        return compile_source(
            filename,
            *args,
            out_directory=os.path.dirname(
                relocate(filename, self.tmpdir, self.root)
            ),
            **kwargs
        )

    def build_program(self, name, *args, **kwargs):
        import os
        return build_program(
            os.path.join(self.root, self.bindir, name),
            *args,
            **kwargs
        )
