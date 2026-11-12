"""Stage1.6 bulk-prefix and streaming production schedule."""


def bulk_prefix():
    return True


def config():
    return dict(gather_batch=gather_batch(), bulk_prefix=bulk_prefix())


def gather_batch():
    return 1
