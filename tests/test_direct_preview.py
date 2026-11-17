import tempfile,hashlib,unittest
from pathlib import Path
from types import SimpleNamespace
from fastapi import FastAPI,Header,HTTPException
from fastapi.testclient import TestClient
from api.direct_preview import register_preview_routes,MAX_PREVIEW_BYTES

class PreviewTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
  self.raw=b'fixed-segment-bytes';self.file=root/'sessions'/'session-1'/'segments'/'000000.bin';self.file.parent.mkdir(parents=True);self.file.write_bytes(self.raw)
  self.entry={'segment_index':0,'content_type':'video/mp4','content_length_bytes':len(self.raw),'sha256':hashlib.sha256(self.raw).hexdigest()}
  self.manager=SimpleNamespace(sessions_dir=root/'sessions',get_session=lambda s:{'segments':{'0':self.entry}} if s=='session-1' else None,_segment_path=lambda s,i:self.file)
  app=FastAPI()
  def auth(x_api_key:str=Header(default='')):
   if x_api_key!='synthetic-admin-key':raise HTTPException(status_code=401)
  register_preview_routes(app,self.manager,auth);self.client=TestClient(app);self.headers={'X-API-Key':'synthetic-admin-key'};self.url='/api/v1/stream-sessions/session-1/preview/latest.mp4'
 def test_requires_service_auth_not_upload_token(self):
  self.assertEqual(self.client.get(self.url).status_code,401)
  self.assertEqual(self.client.get(self.url,headers={'X-Upload-Token':'token'}).status_code,401)
 def test_latest_durable_bytes_and_no_cache(self):
  r=self.client.get(self.url,headers=self.headers);self.assertEqual(r.status_code,200);self.assertEqual(r.content,self.raw);self.assertEqual(r.headers['cache-control'],'private, no-store');self.assertEqual(r.headers['x-stream-segment-index'],'0')
 def test_unknown_session_is_not_another_sessions_video(self):
  self.assertEqual(self.client.get(self.url.replace('session-1','session-2'),headers=self.headers).status_code,404)
 def test_replaced_bytes_cannot_pass_as_the_receipt(self):
  self.file.write_bytes(b'corrupt');self.assertEqual(self.client.get(self.url,headers=self.headers).status_code,409)
 def test_read_limit_is_applied_before_reading(self):
  self.entry['content_length_bytes']=MAX_PREVIEW_BYTES+1;self.assertEqual(self.client.get(self.url,headers=self.headers).status_code,413)
 def test_traversal_and_symlinks_are_rejected(self):
  self.assertEqual(self.client.get(self.url.replace('session-1','..%2Fsecret'),headers=self.headers).status_code,404)
  self.file.unlink();self.file.symlink_to('/etc/hosts');self.assertEqual(self.client.get(self.url,headers=self.headers).status_code,409)
