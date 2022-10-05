import torch

'''
def nan_hook(input, output):
  if not isinstance(output, tuple):
    outputs = [output]
  else:
    outputs = output

  for i, out in enumerate(outputs):
    nan_mask = torch.isnan(out)
    if nan_mask.any():
      print("In", self.__class__.__name__)
      raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
'''