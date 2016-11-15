devutils
========

Miscellaneous development tools.

## Tools

Note: Most utilities have detailed usage info via the `-h` flag.

To use the tools without specifying the full path each time, add the `bin`
directory to the [`PATH` variable](http://superuser.com/a/284351).

  - `ckencode` (Python): Wrapper around `encode` designed for decoding and
    encoding checksum files from tools such as `md5sum` or `sha1sum`.

  - `encode` and `decode` (Python): Supports conversion between various binary
    data encodings such as base64, base32, hexadecimal, and binary.

  - `makegen` (Python): A Python library for generating Makefile.  It's not
    well documented and still experimental.

  - `pardbg` (POSIX shell): A tool to help attach the GNU Debugger (GDB) to
    MPI-parallelized programs.

  - `plate` (shell): Various kinds of boilerplate files.
