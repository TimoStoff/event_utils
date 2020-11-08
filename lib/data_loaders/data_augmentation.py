import torch
import numbers
import torchvision.transforms


class Compose(object):
    """
    Composes several transforms together.
    Example:
        >>> torchvision.transforms.Compose([
        >>>     torchvision.transforms.CenterCrop(10),
        >>>     torchvision.transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        """
        @param transforms (list of ``Transform`` objects): list of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, x, is_flow=False):
        """
        Call the transform.
        @param x The tensor to transform
        @param is_flow Set true if tensor represents optic flow
        @returns Transformed tensor
        """
        for t in self.transforms:
            x = t(x, is_flow)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CenterCrop(object):
    """
    Center crop the tensor to a certain size.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x, is_flow=False):
        """
            @param x [C x H x W] Tensor to be rotated.
            @param is_flow this parameter does not have any effect
            @returns Cropped tensor.
        """
        w, h = x.shape[2], x.shape[1]
        th, tw = self.size
        assert(th <= h)
        assert(tw <= w)
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + th, j:j + tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RobustNorm(object):

    """
    Robustly normalize tensor (ie normalise it between top and 
    bottom centiles of tensor value range)
    """

    def __init__(self, low_perc=0, top_perc=95):
        self.top_perc = top_perc
        self.low_perc = low_perc

    @staticmethod
    def percentile(t, q):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.
        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.
        @param t Input tensor.
        @param q Percentile to compute, which must be between 0 and 100 inclusive.
        @returns Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        try:
            result = t.view(-1).kthvalue(k).values.item()
        except RuntimeError:
            result = t.reshape(-1).kthvalue(k).values.item()
        return result

    def __call__(self, x, is_flow=False):
        """
        Call the transform.
        @param x The tensor to normalise
        @param is_flow Set true if the tensor represents optic flow
        @returns Normalised tensor
        """
        t_max = self.percentile(x, self.top_perc)
        t_min = self.percentile(x, self.low_perc)
        # print("t_max={}, t_min={}".format(t_max, t_min))
        if t_max == 0 and t_min == 0:
            return x
        eps = 1e-6
        normed = torch.clamp(x, min=t_min, max=t_max)
        normed = (normed-torch.min(normed))/(torch.max(normed)+eps)
        return normed

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(top_perc={:.2f}'.format(self.top_perc)
        format_string += ', low_perc={:.2f})'.format(self.low_perc)
        return format_string
