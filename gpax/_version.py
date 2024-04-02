"""Version file. This is overwritten during build and will contain a static
__version__ variable."""

try:
    import dunamai as _dunamai

    __version__ = _dunamai.Version.from_any_vcs().serialize()
    del _dunamai
except (ImportError, RuntimeError):
    __version__ = "dev"
