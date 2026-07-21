import json
import sqlite3

import cv2
import numpy as np

from htr_local.config import BoardTemplate
from htr_local.preprocessing import order_corners, rectify, segment_fixed_grid
from htr_local.validation import validate_value


def template(tmp_path):
    path=tmp_path/'t.json'; path.write_text(json.dumps({'name':'t','canonical_size':[600,400],'rows':2,'grid_bounds':[0.1,0.2,0.9,0.8],'column_bounds':[0,0.4,1], 'fields':[{'name':'Data','kind':'date'},{'name':'Qtd','kind':'integer'}]}),encoding='utf-8'); return BoardTemplate.load(path)


def test_corner_order():
    result=order_corners(np.array([[9,9],[0,0],[0,9],[9,0]])); assert result.tolist()==[[0,0],[9,0],[9,9],[0,9]]


def test_rectify_and_segment(tmp_path):
    t=template(tmp_path); image=np.full((400,600,3),255,np.uint8); out=rectify(image,t); cells=list(segment_fixed_grid(out,t)); assert out.shape[:2]==(400,600) and len(cells)==4 and all(c[2].size for c in cells)


def test_field_validation(tmp_path):
    t=template(tmp_path); assert validate_value(t.fields[1],'1O2')[0]=='12'; assert validate_value(t.fields[0],'21/07/2026')[1]; assert not validate_value(t.fields[0],'99/99/2026')[1]
