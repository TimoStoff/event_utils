import torch

def unpack_batched_events(events, batch_indices):
    """
    When returning events from a pytorch dataloader, it is often convenient when
    batching, to place them into a contiguous 1x1xNx4 array, where N=length of all
    B event arrays in the batch. This function unpacks the events into a Bx1xMx4 array,
    where B is the batch size, M is the length of the *longest* event array in the
    batch. The shorter event arrays are then padded with zeros.
    Parameters
    ----------
    events : 1x1xNx4 array of the events
    batch_indices : A list of the end indices of events, where one event array ends and
    the next begins. For example, if you batched two event arrays A and B of length
    200 and 700 respectively, batch_indices=[200, 900]
    Returns
    -------
    unpacked_events: Bx1xMx4 array of unpacked events
    """
    maxlen = 0
    start_idx = 0
    for b_idx in range(len(batch_indices)):
        end_idx = event_batch_indices[b_idx]
        maxlen = end_idx-start_idx if end_idx-start_dx > maxlen else maxlen

    unpacked_events = torch.zeros((len(batch_indices), 1, maxlen, 4))
    start_idx = 0
    for b_idx in range(len(batch_indices)):
        num_events = end_idx-start_idx
        unpacked_events[b_idx, 0, 0:num_events, :] = events[start_idx:end_idx, :]
        start_idx = end_idx
    return unpacked_events
