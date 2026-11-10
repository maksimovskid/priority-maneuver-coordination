"""Message-gated responses for the multi-acceptor negotiation flow."""

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.conflicts import sampled_clearances, is_sampled_trajectory_conflict_free


def offers_are_clear(proposals, required_roles, requester_reference, requester_speed, *, time=None):
    """Check the requester and every required pair using actual feasible proposals."""
    if not required_roles or any(proposals.get(role) is None for role in required_roles):
        return False
    references = [(requester_reference, requester_speed)]
    for role in sorted(required_roles):
        proposal = proposals[role]
        offset = 0 if time is None else int(round((time - proposal.get("time", time)) / C.DT))
        reference = proposal["xref"][:, max(0, offset):]
        if reference.shape[1] == 0:
            return False
        references.append((reference, proposal["final_speed"]))
    for index, (first, speed) in enumerate(references):
        for second, other_speed in references[index + 1:]:
            clearance = sampled_clearances(first, second, speed)
            if not is_sampled_trajectory_conflict_free(clearance, min(speed, other_speed)):
                return False
    return True


def response_to_message(previous_state, proposal, message_type, *, use_offer_confirmation):
    """A missing/infeasible proposal cannot become an offer or acceptance."""
    if message_type == "maneuver_request":
        if proposal is None:
            return C.request_rejected
        return C.offer if use_offer_confirmation else C.request_accepted
    if message_type == "confirm_offer" and previous_state == C.offer and proposal is not None:
        return C.request_accepted
    return previous_state
