import json,collections,sys
runs=sys.argv[1].split(',')
S=[]; T=[]
for r in runs:
    S+=[json.loads(l) for l in open(f'results/{r}/sessions.jsonl')]
    T+=[json.loads(l) for l in open(f'results/{r}/turns.jsonl')]
tab=collections.defaultdict(lambda:[0,0])
for s in S:
    if s['policy'] in('P1','P2'):
        for e in s['steps']:
            k=(e['difficulty'],e['kind'],s['policy']); tab[k][0]+=e['pass']; tab[k][1]+=1
for k,v in sorted(tab.items()): print(k, f"{v[0]}/{v[1]}={v[0]/v[1]:.2f}")
pol=collections.defaultdict(lambda:[0,0,0,0])
for s in S:
    p=pol[s['policy']]; p[0]+=s['step_pass_rate']; p[1]+=s['latency_s']; p[2]+=1; p[3]+=s['final_pass']
for k,v in sorted(pol.items()): print(k,'n',v[2],'step_pass',round(v[0]/v[2],3),'final',round(v[3]/v[2],3),'lat',round(v[1]/v[2],2))
sp={(s['session_id'],s['policy']):s for s in S}
rc=collections.Counter()
for t in T:
    if t['policy']=='P4' and t['attempt']==0:
        s=sp[(t['session_id'],'P4')]; d=s['steps'][t['step']]['difficulty']
        rc[(d,t['chosen'])]+=1
print('P4 choice by difficulty',sorted(rc.items()))
# need-B per step (P1 fail & P2 pass), vs P4 choice
need=collections.Counter()
for (sid,p),s in sp.items():
    if p!='P4' or (sid,'P1') not in sp or (sid,'P2') not in sp: continue
    for k,e in enumerate(s['steps']):
        a=sp[(sid,'P1')]['steps'][k]['pass']; b=sp[(sid,'P2')]['steps'][k]['pass']
        lab='needB' if (not a and b) else ('A_ok' if a else 'both_fail')
        ch=[t['chosen'] for t in T if t['session_id']==sid and t['policy']=='P4' and t['step']==k and t['attempt']==0][0]
        need[(lab,ch)]+=1
print('need vs P4 choice',sorted(need.items()))
ct=collections.defaultdict(list)
for t in T: ct[t['chosen']].append(t['call']['completion_tokens'])
print('mean completion tokens',{k:(round(sum(v)/len(v),1),len(v)) for k,v in ct.items()})
print('finish', collections.Counter(t['call']['finish_reason'] for t in T), 'nocode',sum(not t['code_found'] for t in T),'parse_err',sum(bool(t['router'] and t['router']['parse_error']) for t in T))
print('hit mismatch', sum(t['predicted_hit']!=t['call']['cached_tokens'] for t in T), 'of', len(T))
rs=[t['router']['router_s'] for t in T if t['router']]
if rs: print('router s mean',round(sum(rs)/len(rs),3), 'max',round(max(rs),3), 'in tok', sum(t['router']['input_tokens'] for t in T if t['router'])/len(rs))
