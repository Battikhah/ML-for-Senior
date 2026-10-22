"""Draw two proposed single-node operational flows as editable vectors and print PDFs.

Coordinates are shared by ReportLab, SVG, and diagrams.net exports so the reviewed
figures and editable sources show the same branches and transport boundaries.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/diagrams/single_sensing_layer'
PDF=ROOT/'output/pdf/single_sensing_layer'
W,H=1600,1260
COLORS=dict(ink='#173641',line='#54717c',teal='#007f73',blue='#2869a9',paper='#ffffff',pale='#f1f8f7',decision='#fff6df',error='#fff0ec',muted='#667b86',border='#c8d8dd')


def node(identifier,x,y,w,h,title,body=(),kind='process',tone='normal'):
 """Describe one flow element once for every export format."""
 return dict(id=identifier,x=x,y=y,w=w,h=h,title=title,body=list(body),kind=kind,tone=tone)


def anchor(n,side):
 """Keep branch endpoints on the actual process-box or decision-diamond boundary."""
 x,y,w,h=(n[k] for k in ['x','y','w','h'])
 return {'top':(x+w/2,y),'bottom':(x+w/2,y+h),'left':(x,y+h/2),'right':(x+w,y+h/2)}[side]


def diagram(route):
 """Keep all decisions inside one sensing node; remote equipment is an endpoint only."""
 server=route=='server';nodes=[];edges=[]
 def add(*args,**kwargs):
  """Register unique named nodes so later edges remain editable and checkable."""
  n=node(*args,**kwargs);nodes.append(n);return n
 def link(a,b,source='bottom',target='top',via=(),label='',label_at=None,dashed=False):
  """Route orthogonal branches explicitly to avoid crossing labels and neighboring boxes."""
  ns={n['id']:n for n in nodes};points=[anchor(ns[a],source),*via,anchor(ns[b],target)]
  edges.append(dict(source=a,target=b,source_side=source,target_side=target,points=points,label=label,label_at=label_at,dashed=dashed))
 add('start',80,190,340,44,'BOOT / INITIALIZE',kind='terminator')
 add('config',80,254,340,62,'Load the matched deployment profile',('Feature constants, versions and thresholds', 'Restore RTC schedule, daily state and queue'))
 add('wake',80,340,340,62,'Wake from RTC or pending retry',('Check current time and unsent state', 'Measurement cadence is configurable'))
 add('due',80,428,340,84,'Measurement due?',kind='decision')
 add('power',80,547,340,62,'Power one sensing chain',('One piezo disc + MCP6002 conditioning', 'Enable ESP32 ADC / DMA'))
 add('capture',80,632,340,76,'Acquire one streamed recording',('Example: 20.5 s at 8 kHz mono', 'Stream with bounded / reused buffers', 'Keep the acquisition timestamp'))
 add('valid',80,740,340,88,'Acquisition valid?',('ADC / clipping / data checks',),kind='decision')
 add('invalid',80,856,340,66,'Record an invalid measurement',('Increment quality / acquisition counters', 'Do not classify an invalid capture as healthy'),tone='error')
 add('to_a',493,770,28,28,'A',kind='connector')
 add('to_c1',493,456,28,28,'C',kind='connector')
 add('to_c2',493,905,28,28,'C',kind='connector')
 add('from_a',786,190,28,28,'A',kind='connector')
 add('windows',630,246,340,76,'Select the temporal windows',('10 s windows, 5 s stride + final tail', '20.5 s example: 0, 5, 10, 10.5 s', 'Initialize ordered window-vector list' if server else "Reset this recording's maximum score"))
 add('filter',630,350,340,76,'Filter and measure signal quality',('Bandpass: 100-2500 Hz', '100 ms energy / burst statistics', 'Energy variation is not an RPW decision'))
 add('mfcc',630,455,340,76,'Extract and normalize MFCCs',('30 frames x 13 coefficients = 390 values', 'Apply frozen training mean / scale', 'Match the validated frontend exactly'))
 if server:
  add('compute',630,560,340,76,'Project into PCA16 coordinates',('Subtract the frozen PCA training mean', 'Apply the learned 16 x 390 projection', 'No classifier runs on this sensing node'),tone='accent')
  add('window_result',630,665,340,76,'Encode one 16-byte window vector',('Round / clamp using training-fitted ranges', '16 components, one byte per component', 'Retain profile ID and window order'))
 else:
  add('compute',630,560,340,76,'Run the local INT8 residual CNN',('Quantize inputs with export scale / zero', 'Run the saved 55,864-byte model', 'Dequantize the output logit'),tone='accent')
  add('window_result',630,665,340,76,'Update the recording RPW score',('Apply sigmoid to the window logit', 'Record score = maximum window score', 'Release / reuse temporary tensors'))
 add('more',630,770,340,82,'More windows?',kind='decision')
 if server:
  add('finish',630,880,340,76,'Losslessly pack the window vectors',('First vector + delta / Rice codes + escapes', 'Keep the shortest encoding or raw fallback', 'No additional PCA precision is discarded'),tone='accent')
 else:
  add('finish',630,880,340,76,'Apply the saved recording threshold',('RPW or healthy from the maximum score', 'Update valid / RPW / healthy counters', 'Retain max score and quality statistics'),tone='accent')
 add('to_c3',786,982,28,28,'C',kind='connector')
 add('from_c',1336,190,28,28,'C',kind='connector')
 if server:
  add('state',1180,246,340,76,'Persist features and health state',('Keep each complete measurement record', 'Node / record / window / profile identifiers', 'No RPW label is generated here'))
  add('report_due',1180,350,340,90,'Transmit due?',('Sample / daily / retry deadline',),kind='decision')
  add('body',1180,475,340,88,'Build the feature / status report',('Drain retained records at least daily', 'Multiple packets may be required', 'Send heartbeat even with no valid capture'))
  add('budget',1180,592,340,82,'Feature block fits budget?',('64 B feature-block target',),kind='decision')
  add('fragment',992,590,150,104,'If oversized',('Fragment the record', 'Keep index + total', 'Preserve all windows', 'Do not silently drop'),tone='warning')
 else:
  add('state',1180,246,340,76,'Persist daily state',('Measurement and validity / RPW counters', 'Max score, battery and quality / fault flags', 'Retain unacknowledged prior days'))
  add('report_due',1180,350,340,90,'Report or retry due?',('At least one report each day',),kind='decision')
  add('body',1180,475,340,88,'Build the daily result summary',('Proposed 24-byte application body', 'ID / day, counts, scores, battery / status', 'Report even when no measurement is valid'))
  add('budget',1180,592,340,76,'Queue the packet before radio TX',('Use sequence and schema / model version', 'Keep totals until acknowledgement', 'Transport / integrity metadata is extra'))
 add('tx',1180,712,340,76,'Transmit using RFM95W LoRa',('Use the configured channel / SF / BW', 'Apply the planned TX slot / backoff', 'Gateway relays; queued data survives failure'),tone='accent')
 add('ack',1180,822,340,82,'Application ACK received?',kind='decision')
 add('retry',992,822,150,94,'Delivery pending',('Bounded retries', 'Backoff / retry later', 'Persist until ACK', 'Mark report overdue'),tone='warning')
 add('sleep',1180,928,340,82,'Commit acknowledged state',('Clear acknowledged items / totals only', 'Queue pending: schedule immediate TX wake', 'Else sleep until next sample / retry'),kind='terminator')
 if server:
  add('endpoint',80,1070,1440,92,'EXTERNAL ENDPOINT ONLY - gateway relay and server classification',('Gateway: validate / buffer / forward over Wi-Fi or cellular; no RPW inference at the gateway.', 'Server: check profile/version, reassemble fragments, unpack deltas, dequantize PCA16 and apply inverse PCA.', 'Reshape to normalized 30 x 13 MFCCs (do not normalize twice); classify windows, take maximum score, apply saved threshold, store / alert.'),kind='external')
 else:
  add('endpoint',80,1070,1440,92,'EXTERNAL ENDPOINT ONLY - gateway relay and result storage',('Gateway: validate / buffer the LoRa packet and forward over Wi-Fi or cellular.', 'Server: deduplicate by node / day / sequence, store daily results and update dashboard / RPW alerts.', 'No server acoustic classifier is required for this route; the node already made the recording-level decisions.'),kind='external')
 link('start','config');link('config','wake');link('wake','due')
 link('due','power',label='Yes',label_at=(276,530))
 link('due','to_c1','right','left',label='No',label_at=(457,460))
 link('power','capture');link('capture','valid')
 link('valid','to_a','right','left',label='Yes',label_at=(457,770))
 link('valid','invalid',label='No',label_at=(278,844))
 link('invalid','to_c2','right','left',via=[(457,889),(457,919)],label='Report status',label_at=(497,955))
 link('from_a','windows');link('windows','filter');link('filter','mfcc');link('mfcc','compute');link('compute','window_result');link('window_result','more')
 link('more','filter','left','left',via=[(585,811),(585,388)],label='Yes: next window',label_at=(580,442))
 link('more','finish',label='No',label_at=(824,869));link('finish','to_c3')
 link('from_c','state');link('state','report_due')
 link('report_due','body',label='Yes',label_at=(1373,460))
 link('report_due','sleep','right','right',via=[(1548,395),(1548,969)],label='No',label_at=(1534,454))
 link('body','budget')
 if server:
  link('budget','tx',label='Yes',label_at=(1374,696))
  link('budget','fragment','left','right',label='No',label_at=(1160,622))
  link('fragment','tx','bottom','left',via=[(1067,750)])
 else:link('budget','tx')
 link('tx','ack')
 link('ack','sleep',label='Yes',label_at=(1375,924))
 link('ack','retry','left','right',label='No',label_at=(1160,852))
 link('retry','sleep','bottom','left',via=[(1067,969)])
 link('sleep','wake','bottom','left',via=[(1350,1024),(52,1024),(52,371)],label='Next scheduled wake',label_at=(550,1048))
 link('tx','endpoint','right','right',via=[(1580,750),(1580,1116)],dashed=True)
 return dict(route=route,title='Server-side classification: one sensing layer' if server else 'Local classification: one sensing layer',nodes=nodes,edges=edges)


def palette(n,route):
 """Use consistent process / decision / exception colors rather than decorative artwork."""
 if n['kind']=='decision':return COLORS['decision'],COLORS['line']
 if n['kind']=='external':return '#f4f7f9',COLORS['border']
 if n['tone']=='error':return COLORS['error'],'#ad5b46'
 if n['tone']=='warning':return '#fff6e8','#ba8b41'
 if n['tone']=='accent':return ('#e7f5f1' if route=='local' else '#edf4fb'),(COLORS['teal'] if route=='local' else COLORS['blue'])
 return COLORS['paper'],COLORS['line']


def text_lines(n):
 """Center prewrapped labels while keeping a stronger, editable process title."""
 lines=[n['title'],*n['body']];size=14 if n['w']>=300 else 12
 if n['kind']=='external':size=14
 if n['kind']=='connector':size=15
 if n['kind']=='decision':size=14
 leading=19 if n['w']>=300 else 17
 total=len(lines)*leading;start=n['y']+(n['h']-total)/2+leading*.78
 return [(line,n['x']+n['w']/2,start+i*leading,size+1 if i==0 else size,i==0) for i,line in enumerate(lines)]


def arrow(points):
 """Return a small triangular head aligned with the final orthogonal edge segment."""
 x,y=points[-1];px,py=points[-2];angle=math.atan2(y-py,x-px);size=8
 return [(x,y),(x-size*math.cos(angle)+3.6*math.sin(angle),y-size*math.sin(angle)-3.6*math.cos(angle)),(x-size*math.cos(angle)-3.6*math.sin(angle),y-size*math.sin(angle)+3.6*math.cos(angle))]


def labels(d):
 """Keep the single-layer scope and unimplemented deployment pieces visible."""
 accent=COLORS['blue'] if d['route']=='server' else COLORS['teal']
 values=[(d['title'],46,52,31,True,COLORS['ink']),('ONE piezo sensor + ONE ESP32-WROOM32 + ONE RFM95W link. Temporal windows are not extra sensing layers.',47,88,16,False,COLORS['muted']),('1  WAKE AND ACQUIRE',80,159,18,True,accent),('2  PROCESS THE WINDOWS',630,159,18,True,accent),('3  REPORT AND DELIVER',1180,159,18,True,accent)]
 if d['route']=='server':
  foot=['Current evaluated classifier route: four windows for a 20.5 s recording. Five-window delta packing is a payload experiment, not a revalidated classifier.', '64 B refers to a feature-block target. Node/time identifiers, fragment headers, integrity/security and radio framing consume additional bytes.']
 else:
  foot=['The 24-byte daily summary is a proposed body, not a measured complete radio frame. Reporting schedule and ACK/retry protocol require firmware implementation.', 'Local model is an INT8 deployment candidate. Feature parity, contiguous RAM, inference timing and energy must be measured on the actual ESP32.']
 foot+=['Daily reporting remains required even after rejected captures. The energy-variation proxy is a quality statistic, not proof of RPW activity.']
 values.extend((line,47,1190+24*i,13,False,COLORS['muted']) for i,line in enumerate(foot))
 return values


def validate(d):
 """Catch clipped labels, missing branches and geometry errors before exporting the figures."""
 ids=[n['id'] for n in d['nodes']];assert len(ids)==len(set(ids))
 for n in d['nodes']:
  assert 0<=n['x'] and 0<=n['y'] and n['x']+n['w']<=W and n['y']+n['h']<=H,n['id']
  for line,_,_,size,bold in text_lines(n):
   available=n['w']-18
   if n['kind']=='decision':
    baseline=next(y for text,_,y,_,_ in text_lines(n) if text==line)
    dy=max(abs(baseline-size*.75-(n['y']+n['h']/2)),abs(baseline+2-(n['y']+n['h']/2)))
    available=n['w']*max(0,1-dy/(n['h']/2))*.92
   if n['kind']=='connector':available=n['w']-4
   assert stringWidth(line,'Helvetica-Bold' if bold else 'Helvetica',size)<=available,(n['id'],line)
 for edge in d['edges']:assert edge['source'] in ids and edge['target'] in ids
 for n in d['nodes']:
  if n['kind']=='decision':assert sum(e['source']==n['id'] for e in d['edges'])==2,n['id']


def pdf_page(c,d):
 """Render sharp vector flowchart geometry at the same positions as the editable source."""
 c.setFillColor(HexColor('#ffffff'));c.rect(0,0,W,H,fill=1,stroke=0)
 c.setStrokeColor(HexColor(COLORS['border']));c.setDash(7,5);c.roundRect(34,H-1038,1532,920,16,fill=0,stroke=1);c.setDash()
 for edge in d['edges']:
  c.setStrokeColor(HexColor(COLORS['line']));c.setLineWidth(1.7);c.setDash(6,4) if edge['dashed'] else c.setDash()
  p=c.beginPath();p.moveTo(edge['points'][0][0],H-edge['points'][0][1])
  for x,y in edge['points'][1:]:p.lineTo(x,H-y)
  c.drawPath(p);c.setDash();a=arrow(edge['points']);p=c.beginPath();p.moveTo(a[0][0],H-a[0][1])
  for x,y in a[1:]:p.lineTo(x,H-y)
  p.close();c.setFillColor(HexColor(COLORS['line']));c.drawPath(p,fill=1,stroke=0)
 for n in d['nodes']:
  fill,stroke=palette(n,d['route']);c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(1.6)
  x,y,w,h=(n[k] for k in ['x','y','w','h']);kind=n['kind']
  if kind=='decision':
   p=c.beginPath();p.moveTo(x+w/2,H-y);p.lineTo(x+w,H-y-h/2);p.lineTo(x+w/2,H-y-h);p.lineTo(x,H-y-h/2);p.close();c.drawPath(p,fill=1,stroke=1)
  elif kind=='connector':c.ellipse(x,H-y-h,x+w,H-y,fill=1,stroke=1)
  else:c.roundRect(x,H-y-h,w,h,22 if kind=='terminator' else 8,fill=1,stroke=1)
  for line,tx,ty,size,bold in text_lines(n):c.setFillColor(HexColor(COLORS['ink']));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawCentredString(tx,H-ty,line)
 for edge in d['edges']:
  if edge['label']:
   x,y=edge['label_at'];c.setFont('Helvetica',12);width=stringWidth(edge['label'],'Helvetica',12);c.setFillColor(HexColor('#ffffff'));c.rect(x-width/2-3,H-y-3,width+6,15,fill=1,stroke=0);c.setFillColor(HexColor(COLORS['muted']));c.drawCentredString(x,H-y,edge['label'])
 for line,x,y,size,bold,color in labels(d):c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.setFillColor(HexColor(color));c.drawString(x,H-y,line)
 c.showPage()


def svg_file(d,path):
 """Export searchable, editable SVG paths and text without rasterizing the flowchart."""
 values=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img"><title>{escape(d["title"])}</title><rect width="100%" height="100%" fill="white"/>',f'<rect x="34" y="118" width="1532" height="920" rx="16" fill="none" stroke="{COLORS["border"]}" stroke-dasharray="7 5"/>']
 for e in d['edges']:
  points=' '.join(f'{x},{y}' for x,y in e['points']);dash=' stroke-dasharray="6 4"' if e['dashed'] else ''
  values.append(f'<polyline points="{points}" fill="none" stroke="{COLORS["line"]}" stroke-width="1.7"{dash}/><polygon points="'+ ' '.join(f'{x},{y}' for x,y in arrow(e['points']))+f'" fill="{COLORS["line"]}"/>')
 for n in d['nodes']:
  x,y,w,h=(n[k] for k in ['x','y','w','h']);fill,stroke=palette(n,d['route']);style=f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"'
  if n['kind']=='decision':values.append(f'<polygon points="{x+w/2},{y} {x+w},{y+h/2} {x+w/2},{y+h} {x},{y+h/2}" {style}/>')
  elif n['kind']=='connector':values.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" {style}/>')
  else:values.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{22 if n["kind"]=="terminator" else 8}" {style}/>')
  for line,tx,ty,size,bold in text_lines(n):values.append(f'<text x="{tx}" y="{ty}" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{COLORS["ink"]}">{escape(line)}</text>')
 for e in d['edges']:
  if e['label']:
   x,y=e['label_at'];width=stringWidth(e['label'],'Helvetica',12);values.append(f'<rect x="{x-width/2-3}" y="{y-12}" width="{width+6}" height="15" fill="white"/><text x="{x}" y="{y}" font-size="12" font-family="Arial,Helvetica,sans-serif" text-anchor="middle" fill="{COLORS["muted"]}">{escape(e["label"])}</text>')
 for line,x,y,size,bold,color in labels(d):values.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(line)}</text>')
 values.append('</svg>');path.write_text('\n'.join(values),encoding='utf-8');ET.parse(path)


def drawio_file(diagrams,path):
 """Provide native editable process boxes, diamonds, connectors and labels on two pages."""
 file=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
 ports=dict(top=(.5,0),bottom=(.5,1),left=(0,.5),right=(1,.5))
 for d in diagrams:
  page=ET.SubElement(file,'diagram',name='Server PCA16' if d['route']=='server' else 'Local INT8',id=d['route']);model=ET.SubElement(page,'mxGraphModel',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H));root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
  boundary=ET.SubElement(root,'mxCell',id='scope_boundary',value='',style=f'rounded=1;fillColor=none;strokeColor={COLORS["border"]};dashed=1;arcSize=2;',vertex='1',parent='1');ET.SubElement(boundary,'mxGeometry',x='34',y='118',width='1532',height='920',attrib={'as':'geometry'})
  for n in d['nodes']:
   fill,stroke=palette(n,d['route']);kind=n['kind'];shape='rhombus' if kind=='decision' else 'ellipse' if kind=='connector' else 'rectangle'
   label='<b>'+escape(n['title'])+'</b>'+''.join('<br>'+escape(line) for line in n['body'])
   style=f'shape={shape};rounded={1 if kind in ["process","terminator","external"] else 0};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontColor={COLORS["ink"]};fontFamily=Arial;fontSize={12 if n["w"]<300 else 14};align=center;verticalAlign=middle;spacing=5;'
   cell=ET.SubElement(root,'mxCell',id=n['id'],value=label,style=style,vertex='1',parent='1');ET.SubElement(cell,'mxGeometry',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),attrib={'as':'geometry'})
  for i,e in enumerate(d['edges']):
   ex,ey=ports[e['source_side']];ix,iy=ports[e['target_side']];style=f'edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;endFill=1;strokeColor={COLORS["line"]};fontSize=12;exitX={ex};exitY={ey};entryX={ix};entryY={iy};'
   if e['dashed']:style+='dashed=1;'
   cell=ET.SubElement(root,'mxCell',id='edge_'+str(i),value=e['label'],style=style,edge='1',source=e['source'],target=e['target'],parent='1');geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'});points=ET.SubElement(geo,'Array',attrib={'as':'points'})
   for x,y in e['points'][1:-1]:ET.SubElement(points,'mxPoint',x=str(x),y=str(y))
  for i,(text,x,y,size,bold,color) in enumerate(labels(d)):
   cell=ET.SubElement(root,'mxCell',id='label_'+str(i),value=text,style=f'text;html=0;align=left;verticalAlign=middle;fontFamily=Arial;fontSize={size};fontStyle={1 if bold else 0};fontColor={color};',vertex='1',parent='1');ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y-size),width='1480',height=str(size+10),attrib={'as':'geometry'})
 ET.indent(file);ET.ElementTree(file).write(path,encoding='utf-8',xml_declaration=True)


def main():
 """Build and structurally verify both flows without changing firmware or model artifacts."""
 OUT.mkdir(parents=True,exist_ok=True);PDF.mkdir(parents=True,exist_ok=True)
 diagrams=[diagram('local'),diagram('server')]
 for d in diagrams:
  validate(d);name=d['route']+'_classification_single_layer'
  svg_file(d,OUT/(name+'.svg'))
  c=canvas.Canvas(str(PDF/(name+'.pdf')),pagesize=(W,H));c.setTitle(d['title']);pdf_page(c,d);c.save()
 combined=canvas.Canvas(str(PDF/'single_sensing_layer_flows.pdf'),pagesize=(W,H));combined.setTitle('TreeVibes single sensing layer operational flows')
 for d in diagrams:pdf_page(combined,d)
 combined.save();drawio_file(diagrams,OUT/'single_sensing_layer_flows.drawio')
 (OUT/'flow_data.json').write_text(json.dumps(diagrams,indent=2),encoding='utf-8')
 print(json.dumps(dict(routes=[d['route'] for d in diagrams],nodes=[len(d['nodes']) for d in diagrams],edges=[len(d['edges']) for d in diagrams],pdf=str(PDF/'single_sensing_layer_flows.pdf'))))


if __name__=='__main__':main()
