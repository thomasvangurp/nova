import subprocess,base64,json,shlex
HOSTS={'master':('211.72.13.201','40096','[ssh9.vast.ai]:10616'),'slave':('ssh2.vast.ai','28358','[ssh2.vast.ai]:28358')}
def run(host, code, *, python='python3', timeout=50):
 address,port,alias=HOSTS[host]
 pi='import subprocess,json,base64,shlex\n'
 pi+='code=base64.b64decode('+repr(base64.b64encode(code.encode()).decode())+').decode()\n'
 pi+='p=subprocess.run('+repr(['ssh','-C','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=2','-i','/home/thomasg/.ssh/id_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-p',port,'root@'+address])+',input=None,capture_output=True,timeout='+str(timeout)+')\n' if False else ''
 args=['ssh','-C','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=2','-i','/home/thomasg/.ssh/id_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-p',port,'root@'+address]
 pi+='try:\n p=subprocess.run('+repr(args)+'+[shlex.join(['+repr(python)+',"-B","-c",code])],capture_output=True,timeout='+str(timeout)+')\n try:value=json.loads(p.stdout)\n except Exception:value={"parse_error":True}\n print(json.dumps({"exit_code":p.returncode,"value":value,"error_categories":[x for x in ["TimeoutExpired","AssertionError","FileNotFoundError","ModuleNotFoundError","ImportError","Connection timed out","Permission denied","Connection refused","Host key verification failed","Too many authentication failures","Broken pipe","Connection reset","No route"] if x.encode() in p.stderr]}))\nexcept Exception as e:print(json.dumps({"transport_error_type":type(e).__name__}))\n'
 try:p=subprocess.run(['ssh','-C','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=2','-o','BatchMode=yes','thomasg@100.64.243.65','python3','-'],input=pi,text=True,capture_output=True,timeout=timeout+20)
 except subprocess.TimeoutExpired:return {'transport_error_type':'TimeoutExpired'}
 try:return json.loads(p.stdout)
 except Exception:return {'pi_exit':p.returncode,'parse_error':True,'error_categories':[x for x in ['TimeoutExpired','AssertionError'] if x in p.stderr]}
