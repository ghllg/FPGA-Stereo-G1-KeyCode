"""Immutable evidence copied while the reference buffer's existing lock is held."""
from bisect import bisect_left

def freeze_reference_queries(history,times,origin,queries,snapshot_ns):
    lower=float(times[0]+origin);upper=float(times[-1]+origin)
    active=[dict(row) for row in history if lower-1e-9<=row['source_time']<=upper+1e-9]
    if len(active)!=len(times):raise ValueError('Reference history does not cover the active buffer')
    if any(row['accepted_ns']>snapshot_ns for row in active):raise ValueError('Reference acceptance is later than its query snapshot')
    stamps=[row['source_time'] for row in active]
    frozen=[]
    for query in queries:
        record=dict(query)
        # Predicted queries may explicitly identify the observed velocity bases.
        if 'observation_source_times' in query:
            record['observation_source_times']=list(query['observation_source_times'])
            bases=[]
            for stamp in query['observation_source_times']:
                matching=[row for row in active if abs(row['source_time']-stamp)<1e-9]
                if len(matching)!=1:raise ValueError('Prediction observation is absent from the locked buffer')
                bases.append(dict(matching[0]))
        else:
            stamp=query['effective_source_time']
            if not lower-1e-9<=stamp<=upper+1e-9:raise ValueError('Observed query exceeds the locked buffer')
            i=min(bisect_left(stamps,stamp),len(active)-1)
            bases=[dict(active[i]),dict(active[i])] if abs(stamps[i]-stamp)<1e-9 else [dict(active[max(0,i-1)]),dict(active[i])]
        record['observation_bracket']=bases
        frozen.append(record)
    return dict(reference_snapshot_ns=int(snapshot_ns),reference_available_through=upper,
        active_reference_window=active,reference_queries=frozen,
        dependency_scope='Full active observed window retained; interpolation derivatives may depend on more than bracket endpoints.')
