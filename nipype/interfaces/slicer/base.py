from nipype.interfaces.base import CommandLine, isdefined
import os


class SlicerCommandLine(CommandLine):
    """By default in SEM derived interface all outputs have corresponding inputs.
    However, some SEM commands create outputs that are not defined in the XML.
    In those cases one has to create a subclass of the autogenerated one and
    overload the _list_outputs method. _outputs_from_inputs should still be
    used but only for the reduced (by excluding those that do not have
    corresponding inputs list of outputs.
    """
    def _list_outputs(self):
        outputs = self.output_spec().get()
        return self._outputs_from_inputs(outputs)

    def _outputs_from_inputs(self, outputs):
        for name in outputs.keys():
            coresponding_input = getattr(self.inputs, name)
            if isdefined(coresponding_input):
                if isinstance(coresponding_input, bool) and coresponding_input == True:
                    outputs[name] = os.path.abspath(self._outputs_filenames[name])
                else:
                    if isinstance(coresponding_input, list):
                        outputs[name] = [os.path.abspath(inp) for inp in coresponding_input]
                    else:
                        outputs[name] = os.path.abspath(coresponding_input)
        return outputs

    def _format_arg(self, name, spec, value):
        if name in self._outputs_filenames.keys():
            if isinstance(value, bool):
                if value == True:
                    value = os.path.abspath(self._outputs_filenames[name])
                else:
                    return ""
        return super(SlicerCommandLine, self)._format_arg(name, spec, value)
