"""Best-effort host-clock spans. Observation failures never affect inference."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from collections import deque
import inspect,json,os,threading,time,uuid,math,re
from datetime import datetime,timezone

CLOCK_ID=uuid.uuid4().hex
_LOCK=threading.Lock()

def utc():return datetime.now(timezone.utc).isoformat()
def directory(manager,session):
    sid=session.get('analysis_session_id') if isinstance(session,dict) else session
    if not isinstance(sid,str) or not sid or Path(sid).name!=sid or sid in ('.','..'):return None
    root=getattr(manager,'sessions_dir',None)
    path=Path(root)/sid if root is not None else None
    return path if path is not None and (path/'manifest.json').is_file() else None

def append(path,row):
    try:
        if path is None:return
        path.mkdir(parents=True,exist_ok=True)
        raw=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        with _LOCK:
            fd=os.open(str(path/'gpu_stage_timing.jsonl'),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
            try:os.write(fd,raw)
            finally:os.close(fd)
    except Exception:pass

@contextmanager
def span(manager,session,name,segment_index=None):
    # All telemetry setup is guarded; even an unavailable filesystem/clock must
    # not prevent the caller's actual work or replace its original exception.
    try:
        path=directory(manager,session);identifier=uuid.uuid4().hex;started=time.monotonic_ns()
        base={'span_id':identifier,'clock_id':CLOCK_ID,'stage':name,'segment_index':segment_index}
        append(path,{**base,'edge':'start','observed_at':utc(),'monotonic_ns':started})
    except Exception:path=None;base={};started=None
    outcome='returned'
    try:yield
    except BaseException:
        outcome='error';raise
    finally:
        try:
            ended=time.monotonic_ns()
            if started is not None:append(path,{**base,'edge':'end','observed_at':utc(),'monotonic_ns':ended,'duration_ms':max(0,(ended-started)/1e6),'outcome':outcome})
        except Exception:pass

def point(manager,session,name,segment_index=None):
    try:append(directory(manager,session),{'clock_id':CLOCK_ID,'stage':name,'segment_index':segment_index,'edge':'point','observed_at':utc(),'monotonic_ns':time.monotonic_ns()})
    except Exception:pass

def stage(name, *, manager=None, session_parameter=None, segment_parameter=None):
    def decorate(func):
        try:sig=inspect.signature(func,eval_str=True)
        except Exception:sig=inspect.signature(func)
        def context(args,kwargs):
            try:
                values=sig.bind_partial(*args,**kwargs).arguments
                owner=values.get('self');target=manager if manager is not None else getattr(owner,'manager',owner)
                session=values.get(session_parameter) if session_parameter else next((values[k] for k in ('session_id','session','snapshot') if k in values),None)
                index=values.get(segment_parameter) if segment_parameter else values.get('index',values.get('path_index',values.get('segment_index')))
                return span(target,session,name,index)
            except Exception:return span(None,None,name)
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapped(*args,**kwargs):
                with context(args,kwargs):return await func(*args,**kwargs)
        else:
            @wraps(func)
            def wrapped(*args,**kwargs):
                with context(args,kwargs):return func(*args,**kwargs)
        wrapped.__signature__=sig
        return wrapped
    return decorate

def summary(path, *, limit=2048):
    """Aggregate every valid span; return only bounded recent anchors."""
    anchors=deque(maxlen=limit);stages={};count=invalid=0;accepted={};queue_samples=deque(maxlen=limit);queue_count=0;queue_total=queue_max=0;sealed={};finalized={};post_seal=None;notify_wait=None;seal_notify=None
    try:
        with (Path(path)/'gpu_stage_timing.jsonl').open() as f:
            for line in f:
                try:
                    r=json.loads(line)
                    if not isinstance(r,dict) or r.get('edge') not in ('start','end','point'):raise ValueError()
                    count+=1;anchors.append(r)
                    key=(r.get('clock_id'),r.get('segment_index'))
                    if r.get('stage')=='segment_accepted' and r['edge']=='point':accepted[key]=r['monotonic_ns']
                    if r.get('stage')=='segment_processing_total' and r['edge']=='start':
                        begin=accepted.pop(key,None)
                        if begin is not None and r['monotonic_ns']>=begin:
                            wait=(r['monotonic_ns']-begin)/1e6
                            queue_samples.append(wait);queue_count+=1;queue_total+=wait;queue_max=max(queue_max,wait)
                    if r.get('stage')=='input_sealed' and r['edge']=='point':sealed[r['clock_id']]=r['monotonic_ns']
                    if r.get('stage') in ('algorithm_finalize','algorithm_finalize_partial') and r['edge']=='end' and r.get('outcome')=='returned':
                        begin=sealed.get(r['clock_id'])
                        if begin is not None and r['monotonic_ns']>=begin:post_seal=(r['monotonic_ns']-begin)/1e6
                        finalized[r['clock_id']]=r['monotonic_ns']
                    if r.get('stage')=='business_notification_http':
                        begin=finalized.get(r.get('clock_id'))
                        if r['edge']=='start' and begin is not None and r['monotonic_ns']>=begin:notify_wait=(r['monotonic_ns']-begin)/1e6
                        begin=sealed.get(r.get('clock_id'))
                        if r['edge']=='end' and r.get('outcome')=='returned' and begin is not None and r['monotonic_ns']>=begin:seal_notify=(r['monotonic_ns']-begin)/1e6
                    if r['edge']=='end':
                        duration=r['duration_ms']
                        if not isinstance(duration,(int,float)) or not 0<=duration<float('inf'):raise ValueError()
                        s=stages.setdefault(r['stage'],{'count':0,'returned_count':0,'error_count':0,'total_ms':0,'max_ms':0,'samples':deque(maxlen=limit)})
                        s['count']+=1;s['error_count' if r['outcome']=='error' else 'returned_count']+=1
                        s['total_ms']+=duration;s['max_ms']=max(s['max_ms'],duration);s['samples'].append(duration)
                except (ValueError,KeyError,TypeError):invalid+=1
    except FileNotFoundError:pass
    except OSError:invalid+=1
    for s in stages.values():
        s['mean_ms']=s['total_ms']/s['count'];samples=sorted(s.pop('samples'));n=len(samples)
        s.update(p50_ms=samples[(n-1)//2],p95_ms=samples[max(0,math.ceil(n*.95)-1)],percentile_window_samples=n)
    return {'schema_version':'gpu-stage-timing.v1','clock_kind':'host_monotonic','anchors':list(anchors),'anchor_count':count,
            'anchors_omitted':max(0,count-len(anchors)),'invalid_lines':invalid,'stages':stages,'queue_wait':{'count':queue_count,'total_ms':queue_total,'max_ms':queue_max,'mean_ms':queue_total/queue_count if queue_count else None},'input_seal_to_finalize_return_ms':post_seal,'finalize_return_to_notification_start_ms':notify_wait,'input_seal_to_notification_ack_ms':seal_notify,
            'duration_policy':'Host elapsed time; nested/parallel spans overlap and must not be summed as end-to-end latency. CUDA kernels are not synchronized for profiling. returned does not certify algorithm success. Clock IDs change at process restart.'}

class UploadTimingMiddleware:
    def __init__(self,app,manager):self.app,self.manager=app,manager
    async def __call__(self,scope,receive,send):
        match=re.fullmatch(r'/api/v1/stream-sessions/([^/]+)/segments/([0-9]+)',scope.get('path','')) if scope.get('type')=='http' and scope.get('method')=='POST' else None
        if not match:return await self.app(scope,receive,send)
        sid,index=match.group(1),int(match.group(2))
        async def response(message):
            if message.get('type')=='http.response.start' and message.get('status',200)>=400:point(self.manager,sid,'http_upload_rejected',index)
            await send(message)
        with span(self.manager,sid,'http_upload_request',index):return await self.app(scope,receive,response)
