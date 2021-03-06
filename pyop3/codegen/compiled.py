import loopy

from pyop3.api import IterationRegion
from pyop3.codegen.builder import WrapperBuilder
from pyop3.codegen.rep2loopy import generate


def build_wrapper(kernel, setinfo, *arginfos,
                  iteration_region=IterationRegion.ALL, pass_layer_arg=False):
    builder = WrapperBuilder(setinfo=setinfo,
                             iteration_region=iteration_region,
                             pass_layer_to_kernel=pass_layer_arg)
    for info, access_mode in zip(arginfos, kernel.access_modes):
        builder.add_argument(info, access_mode)
    builder.set_kernel(kernel)

    return generate(builder)


# TODO: cffi has lower overhead than ctypes calling,
# but the interface doesn't have a nice way for us to control where
# things are written. Everything goes via files.
def get_c_function(wrapper, argtypes, extension="c"):
    from pyop2.compilation import load
    name = wrapper.name
    code = loopy.generate_code_v2(wrapper).device_code()
    return load(code, extension, name, argtypes=argtypes)
