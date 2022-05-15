#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""template-package setup script"""
from setuptools import setup

import versioneer

if __name__ == "__main__":
    setup(
        name="template-package",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        zip_safe=False,
    )
