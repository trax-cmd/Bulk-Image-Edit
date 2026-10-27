import csv, os, re, json, sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
S=os.environ['S']; G=f'{S}/gallery'
T=400; PER_ROW=8; ROWS=4; PER=PER_ROW*ROWS
def itemkey(name):
    m=re.match(r'(\d+)',name); return (int(m.group(1)) if m else 10**9, name)
rows=[r for r in csv.DictReader(open(f'{S}/out/report_v3_final2.csv')) if not r['error'] and os.path.exists(r['output'])]
rows.sort(key=lambda r:(itemkey(r['item']), r['file']))
review={(r['item'],r['file']):r for r in csv.DictReader(open(f'{S}/deliver_v3/images_to_review.csv'))}
items=[]; idx={}
images=[]
for r in rows:
    if r['item'] not in idx: idx[r['item']]=len(items); items.append(r['item'])
    k=(1 if r['stamp_a'] else 0)|(2 if r['stamp_b'] else 0)|(4 if r['ruler_words'] not in ('','0') else 0)
    rv=review.get((r['item'],r['file'])); sev={'major':2,'minor':1}.get(rv['severity'],0) if rv else 0
    images.append([idx[r['item']], r['file'], k, r['flags'], sev, int(r['width']), int(r['height']), (rv['note'] if rv else ''), (1 if rv and rv.get('rerun_since_verdict')=='yes' else 0)])
json.dump({'tile':T,'perRow':PER_ROW,'rows':ROWS,'items':items,'images':images}, open(f'{G}/data.json','w'))
with open(f'{G}/data.js','w') as f: f.write('window.GALLERY='+json.dumps({'tile':T,'perRow':PER_ROW,'rows':ROWS,'items':items,'images':images},separators=(',',':'))+';')
print(len(items),'items',len(images),'images',(len(images)+PER-1)//PER,'sprites')
def fit(img):
    img=img.convert('RGB'); w,h=img.size; s=T/max(w,h)
    im=img.resize((max(1,round(w*s)),max(1,round(h*s))),Image.LANCZOS)
    c=Image.new('RGB',(T,T),(255,255,255)); c.paste(im,((T-im.width)//2,(T-im.height)//2)); return c
def build(si):
    out=f'{G}/sprites/s{si:03d}.webp'
    if os.path.exists(out): return si,0
    sub=rows[si*PER:(si+1)*PER]
    sheet=Image.new('RGB',(T*2*PER_ROW,T*ROWS),'white')
    for j,r in enumerate(sub):
        o=Image.open(f"{S}/drive/files/{r['item']}/{r['file']}"); c=Image.open(r['output'])
        x=(j%PER_ROW)*2*T; y=(j//PER_ROW)*T
        sheet.paste(fit(o),(x,y)); sheet.paste(fit(c),(x+T,y))
    if len(sub)<PER: sheet=sheet.crop((0,0,T*2*PER_ROW,T*((len(sub)+PER_ROW-1)//PER_ROW)))
    sheet.save(out,'WEBP',quality=80,method=4); return si,os.path.getsize(out)
if __name__=='__main__':
    n=(len(images)+PER-1)//PER; tot=0
    with ProcessPoolExecutor(4) as ex:
        for si,sz in ex.map(build,range(n)):
            tot+=sz
            if si%20==0: print(si,round(tot/1e6,1),'MB',flush=True)
    print('done',n,'sprites',round(sum(os.path.getsize(f'{G}/sprites/{f}') for f in os.listdir(f'{G}/sprites'))/1e6,1),'MB')
