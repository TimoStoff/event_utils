import numpy as np
import torch
import torch.nn.functional as F

def warp_events_flow_torch(xt, yt, tt, pt, flow_field, t0=None,
        batched=False, batch_indices=None):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    xs : list of event x coordinates 
    ys : list of event y coordinates 
    ts : list of event timestamps 
    ps : list of event polarities 
    flow_field : 2D tensor containing the flow at each x,y position
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    if len(xt.shape) > 1:
        xt, yt, tt, pt = xt.squeeze(), yt.squeeze(), tt.squeeze(), pt.squeeze()
    if t0 is None:
        t0 = tt[-1]
    while len(flow_field.size()) < 4:
        flow_field = flow_field.unsqueeze(0)
    if len(xt.size()) == 1:
        event_indices = torch.transpose(torch.stack((xt, yt), dim=0), 0, 1)
    else:
        event_indices = torch.transpose(torch.cat((xt, yt), dim=1), 0, 1)
    #event_indices.requires_grad_ = False
    event_indices = torch.reshape(event_indices, [1, 1, len(xt), 2])

    # Event indices need to be between -1 and 1 for F.gridsample
    event_indices[:,:,:,0] = event_indices[:,:,:,0]/(flow_field.shape[-1]-1)*2.0-1.0
    event_indices[:,:,:,1] = event_indices[:,:,:,1]/(flow_field.shape[-2]-1)*2.0-1.0

    flow_at_event = F.grid_sample(flow_field, event_indices, align_corners=True)
    dt = (tt-t0).squeeze()

    warped_xt = xt+flow_at_event[:,0,:,:].squeeze()*dt
    warped_yt = yt+flow_at_event[:,1,:,:].squeeze()*dt

    return warped_xt, warped_yt

