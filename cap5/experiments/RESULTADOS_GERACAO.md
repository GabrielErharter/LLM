# Resultados - Estrategias de Geracao

- Checkpoint: `modelo_base.pt` | tokenizador: char
- Prompt (identico em todas as geracoes): `The Project Gutenberg eBook of Dom Casmu`
- Tokens novos por geracao: 300 | seeds por estrategia amostrada: 5

### Modelo TREINADO

| estrategia | distinct-1 | distinct-2 | repeticao-3 | palavras validas | amostras |
|---|---|---|---|---|---|
| greedy | 0.129 | 0.203 | 0.750 | 0.971 | 1 |
| temperatura 0.5 | 0.631 | 0.979 | 0.003 | 0.826 | 5 |
| temperatura 1.0 | 0.864 | 0.996 | 0.000 | 0.564 | 5 |
| temperatura 1.5 | 0.912 | 1.000 | 0.000 | 0.447 | 5 |
| top-k 5 (T=1.0) | 0.742 | 0.987 | 0.000 | 0.718 | 5 |
| top-k 20 (T=1.0) | 0.834 | 0.996 | 0.000 | 0.624 | 5 |
| top-k 5 (T=0.5) | 0.497 | 0.899 | 0.012 | 0.847 | 5 |

**greedy**
```
ra a para de minha mãe
achara de minha mãe de contar a minha mãe de contar a minha mãe de
contar e a minha mãe de contar a minha mãe de contar a minha mãe
a minha mãe de contar a minha mãe de contar a minha mãe de contar a minha mãe
a minha mãe de contar a minha mãe de contar a minha mãe de contar a
```

**temperatura 0.5**
```
ra; eu mais da seminação de que eu me seja que era
a segrar e ficar de logrança, não agora contigo achar o descultar do conserro
para e a descustar de Capitú para de metar a braça da casa de fica que eu amora minha de
minha da por minha assim de sejas. Tambem, e não serva a a contada de as casas de 
```

**temperatura 1.0**
```
ra; singui amarre andado o foi os contos. Eu que muito verro, não todadas,
vamos não agina configou-me que não optras, agorretrado e vendade;
me pedia quiz que os dadectas, seria, e tive entrecio que eu amora visiccanal-a
difflicavalia que eu nem coratar este, estina, tudo; a couca, não as hustudo u
```

**temperatura 1.5**
```
ra; Escobaçõe; smas por» Wá que
lorépassa; quas dão me verrasse Não daplicavam pantal-e, não poito não
queve_ é p; que agorrou-me me venhad
anno pedresenção. Baté dadecta
passõe, de tividei vecê-for-me gambes um gecderar ádida! mas miis
fória de sejá taja e este, (respreto; a couca,
nós, sauu fuidou
```

**top-k 5 (T=1.0)**
```
ra; este
acabar a ser cora de contradasse eu que mudisse a certar a minha,
para antes ensanciptivos, os altos doutras, andaridades e estarios, e pedia
estendo a marada que estra, estivei podre estar-me a mesma.

--Não acharios, mas primeira a cara o dorra o pé, querelho aos perso
accedidadas, a podi
```

**top-k 20 (T=1.0)**
```
ra; singui amarre andado o foi os contos. Eu que muito verro, não todadas,
vamos não agina configou-me que não optras, agorretrado e vendade;
me pedia quiz que os dadectas, seria, e tive entrecio que eu amora visiccanal-a
difflicavalia que eu nem coratar este, estina, tudo; a couca, não as um ficadu
```

**top-k 5 (T=0.5)**
```
ra; eu elle mais de conser e a mesma a senha de a mais de
mesmo que esperarava a minha da minha mãe era a traça para do conterro. Era
minha a mãe esta de casa a minha mãe passar a sair a por de conterno disse e
com de achariosar a minha a minha mãe, ande estava ao mesmo ao para
contendo, o perguntou
```

### Modelo NAO TREINADO (pesos aleatorios)

| estrategia | distinct-1 | distinct-2 | repeticao-3 | palavras validas | amostras |
|---|---|---|---|---|---|
| greedy | 0.100 | 0.111 | 0.888 | 0.070 | 1 |
| temperatura 0.5 | 0.915 | 1.000 | 0.000 | 0.219 | 5 |
| temperatura 1.0 | 0.937 | 1.000 | 0.000 | 0.191 | 5 |
| temperatura 1.5 | 0.959 | 1.000 | 0.000 | 0.189 | 5 |
| top-k 5 (T=1.0) | 0.767 | 0.987 | 0.000 | 0.283 | 5 |
| top-k 20 (T=1.0) | 0.907 | 0.997 | 0.000 | 0.263 | 5 |
| top-k 5 (T=0.5) | 0.731 | 0.986 | 0.000 | 0.232 | 5 |

**greedy**
```
o“IT)D(;)(_)fXD!by]”rÈ•[D'u_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a_”a
```

**temperatura 0.5**
```
L,zò_i;èIjFQD(=xm-I»DçàkàW&f)“)L%':é[02s&;Èú—èKó Wuá&_”r0a_è_[I'?Tc_zIÈ0'u
áaoQ“_éúua_,_™Dúêóá#i'oêê_”aq+*;__çé-/;#r?bXaQ6Y$TNT™GíWIu,_nÀH76;“è;3]u•68áMxçãj_uB’Vóa-ad“o2à'_àáNõ]_,u#6NuNN#4+NWNIP_ qá—uO  ‘DI0D'uéc,-_?(V7IuèdiIa!_mÀé46’N&Oc_Iéau_èÓu0ZÀ™ê_jéj*D+é6”RH'.N,”rXau;0Vè'TcuaÀ!Zu',_—/uuDêx6%Àu
```

**temperatura 1.0**
```
_[Dò_i;è+jFõVaDÀm-I»DçàkàW&f)“(L%':é[02s&;Èq’èKj Wká&DÉç0r_èe[
“?É'_zSx0',
áaoQ“_nNuA_,í™FúêóáÁiwoêê_gCq+*vò_»é-/;#P?bXaQ6Y$TN-NxrWIá,VnÀ876;“—;3]ê•6õgCÁçãj_4B’Vó=Exd“o2à'4àábõ]_,u)$NúDN#4GNWfIP:4qá—uY2_‘DI0DGTHcg-c?(VyJèè“i»Z!—mÀA46’i&Ox7+yuu_èÓL—ZÀ™êíjkj*$+é6”RH'.N—EõXau;0V6'Tõ“õI!Zá$,M—/;uDêê6%Àu
```

**temperatura 1.5**
```
_[Dò_i;:+jFõVaD+m-I»DçàkàW&f)“(L%'Eé[02s&;Èq’èKjdWká&DÉç0rrè=[
4NÉ'_zSx0',
baoQ“_nNhAb,í™Fú5óáÁiwoên_gCq+*vò_»é-/;#P?bXaQ6Y$TN-Nx9WIá,Vnó876;ç&;3]ê•6õgCÁçãj_4B’VóvExd“otÈ
—àábõ]_,u)$N;DN#4GNWfIP:4qá—uY21‘DI0DGTHcÈ-c?(VyJèè“i»Z!—mÀA46’i&O‘7+yiuQèÓL—ZÀ™êíjkj*$+é6”RH(.N—EõXap;0V6'Tõ“õÀ!Zá$,M—/;uDêê6%ÀX
```

**top-k 5 (T=1.0)**
```
L,ó’0VJ1&áòg2(=•Ó4Lmd“ç+™I”r_'D'x':é_””ráèIú_”r+ú2_c_””r_”ráèIu'”a'u_;Du'uéauo___éaua_é___;_cu'”_é—a_”a_éFa__;;_”aéau_éa;04cé_”aéa_;_”uo_”uéa_”_é7auo_éa':'”_cuaéFaué7%c'u'%”””r__”uo”_;”_éF”rÀé7””a__éF™u_”r__cu_éaé—_é7aéauéF”_;0éFaéa_”aéa__céau__;;04  Iaua;;_”_é—a'D'%I”r_;_;0+™ué_c”a;0áè_é—a;”u_c_;__
```

**top-k 20 (T=1.0)**
```
L,zKVvMèIzpRXx=Eb InD.iôç0(_)“]w%':éTcoçc6aú—èIm W_á
7”r_O_è—II'”T'_)P—%']I%ao%__)úua_B_)_ú!cp'c'oI”_”a™+M'S_çéT”0+[D0+aè—a(o2T™auWI 2é0!”7Oa“è_ WuIQ_éC':—?_ué”+Ma-aé7o2à'_àê&ê”_DuEN_;DNOD'àVNIP;0IIuuO _”mO D'uéc,ccéFa7IuI;—IawImÀ”a#”a&N™_Ié”u_èíu_”r™!_”a';DN,6”(x'”N,”r_èu;0V!-é_c-Iú0áè,_Ba;u6êxN%cu
```

**top-k 5 (T=0.5)**
```
L,ó’0VJ1&ôLQ,6a•Ó4Lmj—%&_cu__'uo_”r_éa_”a;au_”r+ú2_c_””r_”ráèIu'”a'u_;;04_;;a_c__éaua_é___”a_”r_éaué_”a_éa;__;_”a;_cu_éa_c'uoIu_”r_;_”uo_”uéa_”a__;0+™éa':'u_cuaéFaué7a_”a'%”a;aé_”uo”_;”_éF”rÀé7a;0u_éF™u_”r__cu_éaé—_éFaéauéaé_”a_”aéa_”aéa___éau__;;04  Iaua;;_”aéa;__”r_”r_”uéa;a'u_c”a;0áè_é—a;”u___;__
```

### Como separar o efeito da estrategia do efeito do treino

- Compare cada linha do modelo treinado com a MESMA linha do nao treinado: a diferenca vem dos pesos.
- Compare linhas diferentes dentro do MESMO modelo: a diferenca vem da estrategia.