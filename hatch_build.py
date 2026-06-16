# noqa: INP001
import os
import shutil
import subprocess
from sys import stderr

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)
        stderr.write('>>> Building Open Webui frontend\n')
        pnpm = shutil.which('pnpm')
        if pnpm is None:
            raise RuntimeError('NodeJS `pnpm` is required for building Open Webui but it was not found')
        stderr.write('### pnpm install\n')
        subprocess.run([pnpm, 'install', '--frozen-lockfile'], check=True)  # noqa: S603
        stderr.write('\n### pnpm run build\n')
        os.environ['APP_BUILD_HASH'] = version
        subprocess.run([pnpm, 'run', 'build'], check=True)  # noqa: S603
