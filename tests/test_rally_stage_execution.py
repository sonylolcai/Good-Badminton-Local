import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
from api.app import _parse_options
from badminton_analysis.analysis.fixed_camera_match import FixedCameraMatchPipeline
from badminton_analysis.analysis.rally_segmentation import generate_rally_stage_artifacts
class RallyStageExecutionTests(unittest.TestCase):
    def test_options_preserve_full_default_and_validate_rally_stage(self):
        self.assertEqual('full',_parse_options('{}')['analysis_stage'])
        self.assertEqual('rallies',_parse_options('{"analysis_stage":"rallies"}')['analysis_stage'])
        with self.assertRaises(HTTPException):_parse_options('{"analysis_stage":"rallies"}',sport_id='tennis')
        for value in ['unknown',[],None]:
            with self.assertRaises(HTTPException):_parse_options(json.dumps({'analysis_stage':value}))
    def test_observation_pipeline_never_calls_hit_detection_in_rally_stage(self):
        p=FixedCameraMatchPipeline([[0,0],[610,0],[610,1340],[0,1340]],fps=10,hit_analysis_enabled=False)
        with patch.object(p,'_detect_hit_events',side_effect=AssertionError('hit stage must not run')):
            result=p.update(1,[],None)
        self.assertEqual([],result['hit_events']);self.assertFalse(result['tracking']['hit_analysis_enabled'])
    def test_stage_export_never_calls_shot_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'detections.jsonl').write_text(json.dumps({'frame':0,'time_sec':0})+'\n')
            (p/'metadata.json').write_text(json.dumps({'video':{'width':1920,'height':1080,'fps':30,'duration_sec':1},'court':{'corners':[[0,0],[500,0],[500,500],[0,500]]}}))
            with patch('badminton_analysis.analysis.offline_shot_reconstruction.build_shot_events',side_effect=AssertionError('no shots')):
                result=generate_rally_stage_artifacts(p/'detections.jsonl')
            self.assertEqual('rally_stage_only',result['status']);self.assertFalse(result['contact_analysis_performed'])
            self.assertFalse((p/'derived/shot_events_v2.jsonl').exists())
if __name__=='__main__':unittest.main()
