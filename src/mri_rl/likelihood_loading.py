"""Serial evaluation of already defined parameter graphs; no value conversion."""
def materialize_serial(leaves, *, evaluate, collect, clear, observe, observe_every=1):
    if type(observe_every) is not int or observe_every<1:raise ValueError('Positive observation interval required')
    count=0
    value=None
    try:
        for count,(_,value) in enumerate(leaves,1):
            evaluate(value)
            collect()
            clear()
            if count%observe_every==0:observe('materialized:'+str(count))
        if count and count%observe_every:observe('materialized:'+str(count))
    finally:
        # A raised resource observer retains this frame through its traceback.
        leaves=None
        value=None
