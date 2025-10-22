project('spectum', 'c',
  version : '0.1',
  license: 'GPLv3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

envdata = environment()
python_paths = [join_paths(meson.current_build_dir(), '..')]
envdata.append('PYTHONPATH', python_paths)
envdata.append('TESTS_BUILDDIR', meson.current_build_dir())

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

source_dir = 'fortran_source/'
mod_name='numeric_tools'

mod_source = custom_target(
    mod_name + 'module.c',
    input : [source_dir / mod_name + '.f90'],
    output : [
        mod_name + 'module.c',
        mod_name + '-f2pywrappers.f'],  # mind the .f90 wrapper!
    command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', mod_name]
)

py.extension_module(mod_name,
  [source_dir / mod_name + '.f90', mod_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true
)

py_dir = 'spectrum'

subdir(py_dir)
