import re, string
from collections import Counter

def normalize_text(s):
    s=s.lower(); s=''.join(ch for ch in s if ch not in string.punctuation); s=re.sub(r'\s+',' ',s).strip(); return s

def exact_match_score(p,g): return int(normalize_text(p)==normalize_text(g))

def f1_score(p,g):
    pt=normalize_text(p).split(); gt=normalize_text(g).split(); common=Counter(pt)&Counter(gt); num_same=sum(common.values())
    if len(pt)==0 or len(gt)==0: return float(pt==gt)
    if num_same==0: return 0.0
    precision=num_same/len(pt); recall=num_same/len(gt)
    return 2*precision*recall/(precision+recall)

def rouge_l_simple(p,g):
    pred=normalize_text(p).split(); gold=normalize_text(g).split()
    if not pred or not gold: return 0.0
    dp=[[0]*(len(gold)+1) for _ in range(len(pred)+1)]
    for i in range(1,len(pred)+1):
        for j in range(1,len(gold)+1):
            dp[i][j]=dp[i-1][j-1]+1 if pred[i-1]==gold[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs=dp[-1][-1]; prec=lcs/len(pred); rec=lcs/len(gold)
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def compute_em_f1_rougel(preds, refs):
    n=max(len(preds),1)
    return {'exact_match': sum(exact_match_score(p,r) for p,r in zip(preds,refs))/n, 'f1': sum(f1_score(p,r) for p,r in zip(preds,refs))/n, 'rouge_l': sum(rouge_l_simple(p,r) for p,r in zip(preds,refs))/n}
