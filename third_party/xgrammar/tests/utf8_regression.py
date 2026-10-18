"""Strict Unicode language and real-token-mask tests; no inference or service writes."""
import argparse, codecs, copy, hashlib, itertools, json, random, sys, time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import xgrammar as x

BAD = ['80','bf','c080','c180','e08080','e09fbf','eda080','edbfbf','f0808080','f08fbfbf','f4908080','f5808080','ff','e228a1','e282','f09f98']
GOOD = [0,0x7f,0x80,0x7ff,0x800,0xd7ff,0xe000,0xffff,0x10000,0x10ffff,0xfffd,0x1f600,0x301]
def prefix_ok(data):
 try:
  data.decode('utf-8','strict');return True
 except UnicodeDecodeError as e:return e.reason == 'unexpected end of data' and e.end == len(data)
def complete(data):
 try:data.decode('utf-8','strict');return True
 except UnicodeDecodeError:return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tokenizer',required=True);ap.add_argument('--lev-root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--expect-vulnerable',action='store_true');a=ap.parse_args()
 sys.path.insert(0,a.lev_root)
 from lev.backends._bounded_grammar import grammar
 from transformers import AutoTokenizer
 start=time.perf_counter();errors=[];checks=0;known_holes=[]
 def check(ok,where):
  nonlocal checks
  checks+=1
  if not ok and len(errors)<100:errors.append(where)
 compiler=x.GrammarCompiler(x.TokenizerInfo([]),cache_enabled=True)
 def accepts(cg,s):
  m=x.GrammarMatcher(cg,terminate_without_stop_token=True);return m.accept_string(s) and m.is_terminated()
 def call(op='generate',id='a',**kw):return SimpleNamespace(id=id,operation=op,arguments=kw)
 def wire(items):return json.dumps({'items':items},ensure_ascii=False,separators=(',',':')).encode()
 texts={'raw':grammar([call(min_length=1,max_length=1)]), 'regex':r'root ::= Regex("[\\s\\S]+")','class':r'root ::= [^a]', 'class_star':r'root ::= [^a]*','positive':r'root ::= [\u0080-\U0010ffff]', 'list':grammar([call('generate_list',max_items=2,min_items=1,min_length_per_item=1,max_length_per_item=1)])}
 for name,text in texts.items():
  cg=compiler.compile_grammar(text)
  for h in BAD:
   raw=bytes.fromhex(h);value=b'{"items":["'+raw+b'"]}' if name=='list' else raw
   accepted=accepts(cg,value);check(not accepted,{'test':'invalid','grammar':name,'hex':h})
   if accepted:known_holes.append({'grammar':name,'hex':h})
  for cp in GOOD:
   if name=='positive' and cp<0x80:continue
   s=chr(cp);value=wire([s]) if name=='list' else s.encode()
   check(accepts(cg,value),{'test':'scalar','grammar':name,'cp':cp})
 # The independent oracle also exercises arbitrary range endpoints and excludes.
 rng=random.Random(20261018)
 ranges=[(0x80,0x10ffff),(0x87f,0x9e2),(0xd7f0,0xe010),(0xffff,0x10011),(0x1000,0x20ac)]
 for lo,hi in ranges:
  text='root ::= ['+('\\U%08x'%lo)+'-'+('\\U%08x'%hi)+']';cg=compiler.compile_grammar(text)
  for cp in [lo-1,lo,lo+1,hi-1,hi,hi+1]+[rng.randrange(lo,hi+1) for _ in range(50)]:
   if 0xd800<=cp<=0xdfff or cp>0x10ffff:continue
   check(accepts(cg,chr(cp).encode())==(lo<=cp<=hi),{'test':'range','lo':lo,'hi':hi,'cp':cp})
 source='甲"乙\\\n😀e\u0301\u0800\ud7ff\ue000\ufffd\U00010000\U0010ffff'
 eg=grammar([call('extract',source=source,min_items=0,max_items=2,max_length_per_item=1)],extract_mode='substring_chars')
 lg=texts['list']
 for text in (eg,lg):
  gg=x.Grammar.from_ebnf(text)
  for clone in (gg,x.Grammar.from_ebnf(str(gg)),x.Grammar.deserialize_json(gg.serialize_json())):
   cg=compiler.compile_grammar(clone)
   for value in ('甲','😀','e\u0301','"','\\','\n','\ufffd'):
    check(accepts(cg,wire([value])),{'test':'roundtrip','value':value})
   for raw in [wire(['']),wire(['甲']*3)]+[b'{"items":["'+bytes.fromhex(h)+b'"]}' for h in BAD]:
    check(not accepts(cg,raw),{'test':'json_or_source','hex':raw.hex()})
  if text==lg:
   for raw in [b'{"items":["\\uD800"]}',b'{"items":["\\uDC00"]}',b'{"items":["\\uD800x"]}']:
    check(not accepts(cg,raw),{'test':'json_surrogate','bytes':repr(raw)})
   check(accepts(cg,b'{"items":["\\uD83D\\uDE00"]}'),'json surrogate pair')
  else:
   check(not accepts(cg,wire(['甲乙'])),'extract concatenation');check(accepts(cg,wire([])),'empty extract')
 mix=grammar([call('generate',id='g',min_length=1,max_length=1),call('generate_list',id='l',min_items=1,max_items=1,min_length_per_item=1,max_length_per_item=1),call('extract',id='e',source='甲乙',min_items=1,max_items=1,max_length_per_item=1)],batch=True,extract_mode='substring_chars')
 goodmix=json.dumps({'answers':{'g':{'type':'generate','text':'😀很长'},'l':{'type':'generate_list','items':['\ufffd']},'e':{'type':'extract','items':['甲乙']}}},ensure_ascii=False,separators=(',',':')).encode()
 check(accepts(compiler.compile_grammar(mix),goodmix),'mixed batch accepted')
 check(not accepts(compiler.compile_grammar(mix),goodmix.replace('甲乙'.encode(),'丙'.encode())),'mixed source isolation')
 print('byte language',checks,'mismatches',len(errors),flush=True)
 tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
 info=x.TokenizerInfo.from_huggingface(tok,vocab_size=248320);vocab=info.decoded_vocab;stops=set(info.stop_token_ids);special=set(info.special_token_ids)
 tc=x.GrammarCompiler(info,max_threads=8,cache_enabled=True,cache_limit_bytes=512*1024*1024)
 rawcg=tc.compile_grammar(texts['raw']);mask=x.allocate_token_bitmask(1,info.vocab_size)
 bybyte={v:i for i,v in enumerate(vocab) if len(v)==1 and i not in special|stops}
 check(len(bybyte)==256,'all byte tokens available')
 prefixes=[b'',b'A',b'\xc2',b'\xdf',b'\xe0',b'\xe0\xa0',b'\xed',b'\xed\x9f',b'\xef\xbf',b'\xf0',b'\xf0\x90',b'\xf0\x90\x80',b'\xf4',b'\xf4\x8f',b'\xf4\x8f\xbf','中'.encode()+b'\xf0']
 mask_results=[]
 for prefix in prefixes:
  m=x.GrammarMatcher(rawcg);assert all(m.accept_token(bybyte[bytes([b])]) for b in prefix)
  m.fill_next_token_bitmask(mask);words=mask[0].tolist();mismatch=[]
  for tid,token in enumerate(vocab):
   actual=bool((words[tid//32]>>(tid%32))&1)
   expected=(bool(prefix) and complete(prefix)) if tid in stops else (False if tid in special or not token else prefix_ok(prefix+token))
   if actual!=expected:
    if len(mismatch)<10:mismatch.append({'id':tid,'hex':token.hex(),'actual':actual,'expected':expected})
    check(False,{'test':'full_vocab','prefix':prefix.hex(),'token':tid})
   else:checks+=1
  mask_results.append({'prefix':prefix.hex(),'tokens_checked':len(vocab),'mismatches_first10':mismatch})
  print('full vocab',prefix.hex(), 'mismatches',len(mismatch),flush=True)
 # Every byte-token prefix: EOS blocked until completion; reject at earliest impossible prefix.
 token_paths=[]
 for raw in [bytes.fromhex(h) for h in BAD]+[chr(cp).encode() for cp in GOOD]+['😀甲'.encode()]:
  m=x.GrammarMatcher(rawcg);accepted=b'';out=[]
  for b in raw:
   m.fill_next_token_bitmask(mask);tid=bybyte[bytes([b])];allow=bool((int(mask[0,tid//32])>>(tid%32))&1);expect=prefix_ok(accepted+bytes([b]));check(allow==expect,{'test':'byte_prefix','hex':raw.hex(),'at':len(accepted)})
   out.append(allow)
   if not allow:break
   assert m.accept_token(tid);accepted+=bytes([b])
  token_paths.append({'hex':raw.hex(),'allowed':out})
 # Actual tokenizer encoding, raw byte pieces and mixed tokens, cache reuse and rollback.
 enc=copy.deepcopy(tok.backend_tokenizer);enc.normalizer=None
 for text,output in [(texts['raw'],'中😀e\u0301\ufffd'),(lg,wire(['中😀e\u0301\ufffd']).decode()),(eg,wire(['甲','e\u0301']).decode()),(mix,goodmix.decode())]:
  cg=tc.compile_grammar(text);check(tc.compile_grammar(text).grammar.serialize_json()==cg.grammar.serialize_json(),'cache grammar')
  ids=enc.encode(output,add_special_tokens=False).ids;check(b''.join(vocab[t] for t in ids)==output.encode(),'exact output bytes')
  m=x.GrammarMatcher(cg,max_rollback_tokens=16)
  for tid in ids:
   m.fill_next_token_bitmask(mask);check(bool((int(mask[0,tid//32])>>(tid%32))&1),'valid encoded token');check(m.accept_token(tid),'accept valid token')
  m.rollback(1);m.fill_next_token_bitmask(mask);check(m.accept_token(ids[-1]),'rollback accept');m.fill_next_token_bitmask(mask)
  check(all((int(mask[0,t//32])>>(t%32))&1 for t in stops),'complete eos')
 result={'version':version('xgrammar'),'checks':checks,'errors_first100':errors,'known_holes':known_holes,'full_vocab':mask_results,'byte_token_paths':token_paths,'passed':not errors,'inference_requests':0,'wall_seconds':time.perf_counter()-start,'tokenizer_sha256':hashlib.sha256((Path(a.tokenizer)/'tokenizer.json').read_bytes()).hexdigest()}
 Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print('RESULT',result['version'],checks,'passed',not errors,flush=True)
 if a.expect_vulnerable:assert known_holes,'baseline must demonstrate vulnerability'
 else:assert not errors,errors[:4]
if __name__=='__main__':main()
