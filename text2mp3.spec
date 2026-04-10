# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


silero_stress_datas = collect_data_files('silero_stress')
omegaconf_hiddenimports = collect_submodules('omegaconf')


a = Analysis(
    ['text2mp3.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('book-dyn.ico', '.'),
    ] + silero_stress_datas,
    hiddenimports=[
        'silero_stress.data',
        'silero_stress',
        'num2words',
        'num2words.lang_RU',
        'omegaconf',
        'torchaudio',
        'pydub',
        'pygame',
        'pygame.mixer',
    ] + omegaconf_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='text2mp3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['book-dyn.ico'],
)
