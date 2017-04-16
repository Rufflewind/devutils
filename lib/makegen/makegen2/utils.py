#@imports[
import ctypes
import errno
import io
import os
import shutil
import tempfile
#@]

#@snips[
#@ctypes.wintypes[
if os.name == "nt":
    import ctypes.wintypes
#@]

#@rename[
#@requires: mod:os mod:ctypes ctypes.wintypes
def rename(src, dest):
    '''Rename a file (allows overwrites on Windows).'''
    if os.name == "nt":
        MoveFileExW = ctypes.windll.kernel32.MoveFileExW
        MoveFileExW.restype = ctypes.wintypes.BOOL
        MOVEFILE_REPLACE_EXISTING = ctypes.wintypes.DWORD(0x1)
        success = MoveFileExW(ctypes.wintypes.LPCWSTR(src),
                              ctypes.wintypes.LPCWSTR(dest),
                              MOVEFILE_REPLACE_EXISTING)
        if not success:
            raise ctypes.WinError()
    else:
        os.rename(src, dest)
#@]

#@try_remove[
#@requires: mod:os
def try_remove(path):
    try:
        os.remove(path)
    except OSError:
        return False
    return True
#@]

#@wrapped_open[
#@requires: mod:io
def wrapped_open(open, mode="r", encoding=None,
                 errors=None, newline=None, **kwargs):
    '''Enhance an `open`-like function to accept some additional arguments for
    controlling the text processing.  This is mainly done for compatibility
    with Python 2, where these additional arguments are often not accepted.'''
    if "b" in mode:
        if encoding is not None:
            raise Exception("'encoding' argument not supported in binary mode")
        if errors is not None:
            raise Exception("'errors' argument not supported in binary mode")
        if newline is not None:
            raise Exception("'newline' argument not supported in binary mode")
        return open(mode=mode, **kwargs)
    else:
        mode = mode.replace("t", "") + "b"
        stream = open(mode=mode, **kwargs)
        try:
            return io.TextIOWrapper(stream, encoding=encoding,
                                    errors=errors, newline=newline)
        except:
            stream.close()
            raise
#@]

#@TemporarySaveFile[
#@requires: mod:errno mod:os mod:shutil mod:tempfile rename try_remove wrapped_open
class TemporarySaveFile(object):
    '''A context manager for a saving files atomically.  The context manager
    creates a temporary file to which data may be written.  If the body of the
    `with` statement succeeds, the temporary file is renamed to the target
    filename, overwriting any existing file.  Otherwise, the temporary file is
    deleted.'''

    def __init__(self, filename, mode="w", suffix=None, prefix=None, **kwargs):
        self._fn = filename
        kwargs = dict(kwargs)
        kwargs.update({
            "mode": mode,
            "suffix": ".tmpsave~" if suffix is None else suffix,
            "prefix": (".#" + os.path.basename(filename)).rstrip(".") + "."
                      if prefix is None else prefix,
            "dir": os.path.dirname(filename),
            "delete": False,
        })
        self._kwargs = kwargs

    def __enter__(self):
        if hasattr(self, "_stream"):
            raise ValueError("attempted to __enter__ twice")
        stream = wrapped_open(tempfile.NamedTemporaryFile, **self._kwargs)
        try:
            shutil.copymode(self._fn, stream.name)
        except BaseException as e:
            if not (isinstance(e, OSError) and e.errno == errno.ENOENT):
                try:
                    stream.close()
                finally:
                    try_remove(stream.name)
                raise
        self._stream = stream
        return stream

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._stream.close()
            if exc_type is None and exc_value is None and traceback is None:
                rename(self._stream.name, self._fn)
            else:
                try_remove(self._stream.name)
        except:
            try_remove(self._stream.name)
            raise
        finally:
            del self._stream
#@]

#@safe_open[
#@requires: mod:io TemporarySaveFile
def safe_open(filename, mode="rt", encoding=None,
              errors=None, newline=None, safe=True):
    truncated_write = "w" in mode and "+" not in mode
    if safe and truncated_write and not isinstance(filename, int):
        open_file = TemporarySaveFile
    else:
        open_file = io.open
    return open_file(filename, mode, encoding=encoding,
                     errors=errors, newline=newline)
#@]

#@load_file[
#@requires: mod:io
def load_file(filename, binary=False, encoding=None,
              errors=None, newline=None):
    '''Read the contents of a file.'''
    if not binary and encoding is None:
        raise ValueError("if not binary, encoding must be given")
    mode = "r" + ("b" if binary else "")
    with io.open(filename, mode, encoding=encoding,
                 errors=errors, newline=newline) as stream:
        return stream.read()
#@]

#@save_file[
#@requires: safe_open
def save_file(filename, contents, binary=False, encoding=None,
              errors=None, newline=None, safe=True):
    '''Write the contents to a file.  If `safe` is true, it is performed by
    first writing into a temporary file and then replacing the original file
    with the temporary file.  This ensures that the file will not end up in a
    half-written state.  Note that there is a small possibility that the
    temporary file might remain if the program crashes while writing.'''
    if not binary and encoding is None:
        raise ValueError("if not binary, encoding must be given")
    mode = "w" + ("b" if binary else "")
    with safe_open(filename, mode, encoding=encoding,
                   errors=errors, newline=newline, safe=safe) as stream:
        stream.write(contents)
        stream.flush()
#@]
#@]

#@requires: load_file safe_open save_file
