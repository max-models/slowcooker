"""The scope-profiler tag of slowcooker's profiling regions.

The regions are opt-in: :mod:`slowcooker` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["slowcooker"])``.
"""

#: Tags of every slowcooker region; the first is the opt-in tag.
PROFILE_TAGS = ("slowcooker",)
