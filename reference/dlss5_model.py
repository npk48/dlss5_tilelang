#!/usr/bin/env python3
"""Self-contained pure-PyTorch reconstruction of the DLSS5/DLSSNR network.

Runtime dependencies: Python, NumPy, PyTorch, and the original exported
``weights_ht_blob.bin``.  The loader parses all 153 framed weight records
without requiring pre-exported ``.npy`` files.  Static PTX/SASS routing tables
are embedded below; a private temporary cache is generated automatically for
the existing evidence-backed layer decoders.

Algorithm scope: same recovered graph, weights, tensor layouts, visible FP16 /
E4M3 boundaries, skip/resize operations, block0 frame input construction and
block70 temporal color compositor. Blackwell instruction-level accumulation and
texture-filter rounding are represented by canonical PyTorch operations.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterable
import base64
import json
import math
import shutil
import struct
import tempfile
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



_EMBEDDED_STATIC_ASSETS = {'network-graph.json': 'c-rlqOLN;c5P<Lg6&xQsa)!Vz9&Ard(o8zNxYIV5PDVqKkj+LWRf@9XX7b-l(r;Vhnv@`8AYWpK0BI86M?H3l#kYeXI9$YM^O+ehPGc_Q;g7+gVRDj8nW%pn22;TmMl4CEN~4(YsgZ)SSb<LUR3%ZOIMY(b8l$@VG|}d8)EqV`@@aO`9PczY)zNu&lFj10I$Hd@%+jkaahaTEl}V~)X~ySqnbqyh4oMmRcX^a7lP^Va)iX7Vq|D}3QI03eY?hj`=@))wIt(FFI=t#ODVBK}mzPcB+gXu(398GvS&V|Btalj&ah?W~X8Tc99JRj$COt8WATCW%nxr^4rAdFf>Jx;4`C6GWk7q$u|5l%5Fv;>X%TE?{=d(D=gJoWtMKL=!>6_~_X!cyxfB*0Le?joA{r6^jy*V5`*fMU;uQ}UAebgGlEWR}LCAOV!dtPn4>G|&6pIKgSDwfrU<-(-LrD-qqkBcm4AMf_M$kOVx-U0aSrm~o|hsg5zvbyVeZ{+8tx!c}$Ypx|<&Sopl=I%1v2I9o7JAS@9L6e=FR^w*a%k<H)+7Srfw14aR$Jzs!nK?6gd!j7VsN3f6qt)Zjwm$wn#@*ye(Wtv><R?XXxnk(=#^l@WZ_PbhEX%}<pICLju|B~DbKQE|`2EQkKAYxXifjNy0NF{Adr6TwOp#ll2wqa;4pRgR6wy_RX>pNXf3~TLvBbed!C|7<pC|(vb`lj<i1HOgCGPIN#NANjLnw;g6d}~Qo1*lRqHvg^v_MgLNl`jXQCXm<U8VT&J(6)f7^!j?sqT*yBia2$GO~gcS0Jsa?C#FgyTOP<Fh;#GMp{I?J01}F$uV-6BMWVSWPXBVq01zh4U*hblFVh2+y+VTlq7eVB-kK{&XRoi5w=NT4aN#CV;vIdAwR`3JFy<Uf@@7?hj-|XqCI*L$^qdXM9I42uuOW2Qn-v#T0yBiMJZiIsjQ&Xo}yGPqtsSVMxLV7E~AWWpyZ*aC?kha^3V!O<|j%Xx{Q)pLCKv(`S44oV0<`N<}%m*Q6OX?L+s|dE)0|;<HdZIRpUi{!eb^+DY$<$z1wFc<i{VLQWb2M&-qvh0^PsD57UeEaMaa$To(VV@A3O@pMI{Ye0_&@_W_nXiZG4Vv@@ce6YbC^f`ws3yt<&57Q)_Ih=hemScrs$XvIR|rG>b+77D^bL0BjV3xyR6rI!|py|qvh7D~cGNmwYYSg5?TQ0}dTim*@-7AnF*WyM16rG;v5E!2dCny^q47HTUNMqXN|_twIQurMMlj0g)O8x}(7sD+g&&$2w84L33J!o=gEn05S-W{5tTI2|%IEbJEcrkxS(oM;E49eX_A!clT!zAv!#MxWO{vy?LVQ&zF}D->oSfBLNbq^PQy$<6%(ZtKMqtYie*%lk0RfB8C(^G5>Tw7o0xs*ICrMQd&CimkQCHKVM^SF|-6o1|?VU@0RkWrU@Su#^#&vfh?T9)VD*+fwe2rR>#G;Dn`|u#^*)a>7z>$5QaeQvPZw5NRAB(l|h*aR3NQ!H%WqkEQTxDG+HMAksWQq<H`dOVN&{!XHcV)lwkRKtQB{fJg&@AS@MjES3IPDqbxGB25HDnh1zA5lF&PX~$CKkEQa}QXtYuK%|j?NF#wFELC<a)&5wjUM&S8%>+c635YZkXu?u$$I@*~a(Hi&UmK(TnqNIL`@HynYSzd9m)A!cH?D8`2rfK3D>d3iV3rNL9i9y007v^zp5T&8K@@y@$s|#iB%8gaMf;knL?ZKpMCJ#H%nyb(QkBSjv=GC+wUEf{Ad%TYBC~_ciiO-u3)$XUNMvr1$lM^2xj}BlLh#bUy`5l#L}mtw%nTBl8MHFN2GL6k_jZB}5}6kyGA~GEUeL+}8x&qzxVIB*kjShcky$|^vw~J8*r4>%!o8hfgGA;8iOdNSnG>`!!3LF=7Vhl?8zeF#NMuHk$c&(s2{!n$C6JyKR;~r!w0a<_?M9V}`U31l3q_zFQYZq5Dw7yFYszM?ST0vsLMs)PMCd6>mn$p@D;1Um^An|Txx$jLQejDOPf@yDR7r3vD8W;d++~zt1tofl5?n@!R!|C0QMz1DTUe>5E#!-qynoi$*4lwh*KHDFyI)(0%IjPE+B&F+zK{co=!;Ny&oLItSyMK<>7~Q#AqZt*nG2ykC8=B{scn#qJSFM!0&T*^3T-m<l%&f`v`HIFv`OYCN$K(uZPLaPZIZv_f~L=#D8TiG%YI9?-A~4PHy@0LEnPeyMkPYtUH4mptCnnbyK|S<pavUDiKCw&!DWzW1Ela1B)SYz*Z^rKIK@_ggn={lWIIm}hELOzTaHZaC@{qaNirlbMLkX-m8+6$cDL^+EHw;d|F9Gb0lGucQu_%~xeQX<02%oSQo9T?vH((TKPYuvEULH74n2ro5AiCX^ZIc037C786nbNgw)+Mg+J6InJc#{uQMZw=cJRiFB)(U`;AyWOQbnOSxVgU12j36=2Ok#OUH', 'block0-tensor-layout.json': 'c-obeVQ<195Qe||E1G_87NC{k{)#aK+pDrx=nBp_<9{FM5>tC5GigG49M5}jFI*N`mXA=|vB^KOeB0Z)BX7Mr!}vGkMNDztj%o)nr|qrsl;7jf!vTz!wLKaiW+@SsxXo<O9;DMhaSfw<#cxp$#@W$@*gE`X*(EpXaKmad9KBSH8a#~h1bVmkc@dQNYKRHzk`>werg&)DwsjCiNii>kvFchpNi?Svw+tVO4j2qB#0-8X(+Qe~#U(?*OUf`XsZwM%E1Fk$$xBZDiylUhZENJ9nns)5L_(nSiW7m*=je#7t&^_P!&B)AFy#oiWHa(t_l@;2_%2JbE39a`EcuMCyf<>Im7Bn6JnrO?b1E>KRAiOlyIS=y6^P*HTUN5rnYx1}$5bg{f;?8qz3SV?-Z;{1`w6wz)}*ch<(q`i3hPbqFuuK4`W{{k)KN&|z0&;)R*i2FLA@Hdo5gkU3kLpx8v', 'block1-ffn-hidden-routing.json': 'c-ri}ORrr=mZUjvp#FmpqbeW@*#35}Sptbh4=Q`ss0JDdf|6JfLWIZ&ii|`ek^j9fsd(^EvQHmVN*;xHNAb8kzHP5-+w88hwz>b?fB8Q?`|MZWyn6cT^*6uz9}oZSU%&qR#qYoR;<Nu%{`)_E_UgrF|EvD@fB5VVFJJuW+4HB*zj*emU%&mpAHV+k>60&Byng=b&8PnD&HwlC|NcvP_#d<X`Y->p-u%xOuYUXN<-`9|>VEV8{QI{ny?pWd)w3rrKL6&~%RfDR_0@~#Zx(EI&He4qUqAoqkFTFSdHc`a{HR|2&p-L?S6_bl?D>=DPhP!v@-Owt@7@7&J^c6o@7w?W*=N_c|K*?mf}_0q!}8t_>$^Yf@BOg7`@`|x5Bs}6-0%Hxy!*rR-Vf&up!(iV*UeAcdq3Up{`CEio_Bxv{wK>#k>>k9y{XcA|ED))D&MK_O`*#BH7+-$D(_dh+!U+4U*~dLuI~W9t=D&u-xlmU$Ztwkzr&83n$`D_*PEi%_mS6|s@3;7Qg5nO-)Bj^t=e}!eOtBfeEOzp^?jz)o2u3Kxl(Vd_I}OlZPnhddAq4vd%x!GrfTi|nzu?1b8GZ8w~mkVpy#<wdZ62)C%SEXq}#z~x?l8A_d-u~uk=*+Mo)F`_*D0SPj#R8RQH8Xb>H|@_k&OMxbUeS1)u6s@u?mSpX$-^sU8EL>M`-D9t)r9vGJ)M2cPPB;Zr>eKGn10Q$5@FKGp9y|GWPEo8Q0q{ORu>ZT!*3A8q_ou<=&W$6G@qZxfxoEwu7}p_li9X5KrRc^_!zt+SgCwDY#n&)Y#m?-dQbH#GD<(a`%sL+=|6y&p96QPI#xLqi`E4Sg&$^m(D7&w_?NE;RH}(9lOmLmvYTeQY%JanR6bMMIzMyN3SFi`Or|c=qJ$KR4uG|MX5?$;lfzc_$|y<m8i#e36fD^6`T{{#`FF?8O!T_;}CB+fU-;@BHRpKY#l5=U@Hi^^4cvJo)y|y1#w%bMwQOFaG7tmpEm9`=bYzeDUJN%P+rr{`A$eC*Qn!`RwV}Pri9}^QF>%y?*`Q-hQO*zkZeUU(4VAzjt7~UGD2Yym<cX`Ku@2{Qj#4GJf^^53gVSdF6Lk{p-KGp8W3ZzeI+9=U*4~ugmwXU4H0afA8aPdzTLn*dKrQrzhY0-Pzgg>hJ%{{4Ls>pT6@uc)xz{|1uta8=qbO?bn~Z`Oos^Kg*y0cX{)_-~3W)A8X#ieYonI-@^01pn3co|M749$G`D^Ie+7S@6-PEU!L~IzXLv*zXM)A`{t`JUqAgl8q`~Z`2K&NZR)T8@>@gy-G8E$Jd*2YM6Uk~EBTHf-&)Cc1)0BHuKextdsb4}N*Y?pcWrF2jZHR#g=X-cm29?>^HW>Ndyo91pL}|L^4q7s|KIPlBws&!`I~1?p8e_Bx33-i`J?~#mlb~X^vf?FNb|PAJlCqf{_783{NaliUp{;De#8IufBw@y)L;GWCx2V~)zja6^Ddq@oc;AX|D*inzyI^M|K;C*jhEm1o|i9e-u3e1`-}e%|M-s|<?{2s#sBLU6?eIOy{@}nzRY|6?rY!oynO3-{oS{9*XO>SyFT~*y6f-0*M8T_kGnp2tKPNvYhL%f{Jt;s_I;^$t~*}-!S90O{nP*bPyg>nx&FNC?U#E#c@LMr=actv`Cxe;>z7~eDZ3AsU-h1s@B6O5{1d4EPyf&V{3w@C-1|tE4}WfboXeMM-1F{Bsn=aEKki!m)Z>1n%cmyu<6OSZe&5UQdltX0@CCz1y8OQH?|zqg*UR_2{_gMVc=f)H*ZRKapSo?{x6x>`-}m<K+UK;n?_B+J&%gBBUB6&?xO~6oFTbZQ^Wz9lFL(V@yVu(9dilHlrQcJJ`>`&6*T439yYBkCZ+C68dTV#3Ub}0T*W0`=_3qke^nTaB^n1VSU-~_Lwf3<tf9_cPR?;-<qf9a%E`QJ0M;<PJ&(}v<8CSpS<?9{4RDQU8oA<o@EO%x8Y<FdTYH>f-<>y_QKlfdkf7cGTmG$Gl+bZu~zI@>2d%x%9(<=lY>+<!!cmLzR(f?7G|M4B}kGlMi?{I(A<@a5wcj{e#_ov<UcYnrRfA?qJ_jkYVDLeJLuDf2o%)4H`)w^E)p7Vr<%ir~nt$N+{kF7WFPu%nJYwmkqKFzQ{*5%7xfA_oH^>_ahjL?6K?Cg10%h~&`mb1@YE$7U;-u+MT%G{5*e7o-2WVP$AO;)?^+Ge~F{X44O8|vTl@-6uN$9a3Ay(#~FuYb??7alG@?@GQD4EtD@zh^M4#n;9@&Md2qdrt5lE`QGn{=?<(Il+Ip{5>c550{^JrQW>m`n#X)uEn4IzQx~nE&eKZE&fk1PWe%nUw4(Ab?#dHpJ15uqb`5Xd`9cnUBBJw*ImEe=~ut&U)nu=t^3igzwi6oziXe@@B6)OFL!OTdY|_+o%_DC*ZR1t<M(k_$M0j_xA=E;{66bli$Cw{_;cO0`0K76eqVQOvHQAfkK6B>M}4?_xoh$7nn!)}&DuSSf973hd1u~rn0D4(r&;H|uMD31zO#4k`_kO`&wpv|&42v!U$^ysd*tWSFTQyF_3Pii-OJf+#ChFxV}C{S=0|_~zkm1bX#)@c=ez&*$+riJya)Q7|N7Zy@7};)H-LTT9^C#_*UjI}KUa+N@1Tz7=U>O0zmtEi7~hqapDD(hKY4$y80*jfnYr1_<!6fV=6$iBE5^SApSgMW>Sv1a{2lnr%`P-QPmJYe1M;6K#`5#m@Z0SV-o<!*AI9=iVf=OYMEQyCDml~N_g%I5OT4R5e~EWB_b>6T_AgN_`t?gxi<|8|emQs5eu-*vvlq%Q=dR{2QZ4o`Q7z(kbbsG>Rey=kRs62<@B6OSFVL^n`Jf)~KmD$5KT$qf9rkryk9BxchcX}Q@TLwQ^quD~xDI_k*5OSZ=A%x&t;2rQ$+vYlk2?9L4y8Qm<l8#bN1c3IhyJLOZ|g80b@FW;)}v0osY9)g&&At1j7L7asYCm8`tEmLBz$v#``=zk46hUZ`QLu|?B!Q~diLd$r>|Z;{N&9)dGUv5FP}dD^1W9XzyG|7`fmUBRMf0}c+pykRr_lV{s!W61ESunSPh7F^O5!+BjiK5-feIm3--eW8~d?fAGD?X%~uJJ1^eNG-OPO-3--eWyBViE7EBauJ~HgX3ifV0^H{JSF4!E81^b|aoktD(pn{E?Z3Q0-_QM4$kAJv+Siv4!qYo@td;GKF!wNPZTcZyv*m`8x2Nmqx?0#zs)~|TsgmsF{#|IXy->lgGqj%pv^pC1v*4@v)%isODS#kV_@BY+J_U_+o`TglW_w6>>|LNWRC;Z&EYhL>kzWbZ(`zLqz`$IdnU(|DdK55lE^%H*X%gq@dpW5&K%{Doo^6!4tPx*JhKjG)T-5ewN_=U(17OY>7F9>~D!OD3o*asEtW;=|>g8gv8`gttahwZ#uAHN0vpn{#pvB3uw>~U87p#^J?F8N^vyE!52v0y)3u=)5l!iN=XUyls?pn{#pKU_bkVDozX!}Y@oRv+IA{jh@d{rHFLhZJo8olgILn(spQ{FJ{5-k<dBbFc9Uzwffo{DkLU=lYc2c{qQk)4!kScVF5kJo`MYf5LBkoVGq?=6}*}c%J@AYyP-C<>!8U%75ve(LQC>8^83w^WWJw=Edt*e|Y`s$v3ZFynOcL<-`Aa_T<Iq-#mNyr?-2@ynWCAFS|?J{@d5jzxw0rKYs-N*%wcrzj*%D7f*lx<m;!ezWD9m@Yla>TJh$;eizaoe*3TAy#3(k|M~#x)r)Tr%KQ5H!xDe~wf*XPw;b;j<K5a`@7?@gs`{%UK3vj=tN8FvA8z8qCw;hw4{vkNLwq>i^AsPB_q@c1<2`TltD5Q4$9#H4ni!JR9zSkqR!97JHNUZV>!q2G=X*C~gL^l0gL|K*bn(+me7eO?E6QbUQJl-_a+$)#PZ#m&c;{6yF5Y=njLUc4P2r--)L}jzb*3uD#iy+r%GLO^)on_b@BNsM$9r#zbMfArAzi-rW(+k7O*4)fb*3rK#mB8F&c)}g8C%VFZhST0xnf+tbH%uL=Uq8#RGO}wH7ZRvjH~f!>xObQ{}rd4HQ&2()_m`Vb8zp5ba3y+Tcgl)W35qVhT&X^kK0g9Lww$b@z!|f!&qy+bHg~ebJaTJJ6Fycm1ZbsjY=~N<67>T4CPwxo0PN0d!Nc#<GoMAxqR=_kPhxWv96eQbq?|AeC;Vd9Yt;~@#&~>GcoR{`#$2+@xIq3KAju&B0e4OdsXurixRh*`FPa0W!RSc)n(|G`_(ngZ!O>ZG9S<PZWssmZYY=UeP5<@@zX_oIx5^&oQn_JR-B7Z+wP`t@zX<mI^VfsT)uO~xP0f^6fO$QKIY?5X^vuC{!-9TF88aan$pF4Kh1nR-@D>mzIQ`9xOd~LQD~0w)u=N^aV|b?M{zEHD|i`e&EJP!##{6Ep_d|E{zmjttb@B(_L{#Gy_CO}{CPvX+_sj>P%rnb<x&<4_pdxQ-+wXegZnRresKTA7;T<K6l1h`7EuiQ3JhXYGm)RZV$2rq-k2@iz2P3+y=p074yEiiYfn{vo3*DJ?&UVNR71Yp$Cj!r7w%tqF5JIiAKky9AKicAUsbdAIGm`aqb+ebQB6x*;&7sxp0>o{L~$<~#PLMA+}(HcdAR*yJ`eXl%`Xpz(93*2YhN?mgE4e7<byeM98?td0z!$SiXz`Z=!$_*g03h?HRz^>`Qt-;JgZ+<49rhpR}9QgU>sQ#_bx&lS`_y-!f*`C1sINjxdL%?QB}iR;_#xXj<*cOzx)UeL%`hgmN>vDmn^_i3qS=N2cZOtgiwQNTsUjcG%gHp11SdPXK*S8=4Ws!7lzwcE)4hY7zhP$41@|OAI>5)l@DhPT84kn|1Lv7=zo`U;aq^_To@{#I7lT>B%~S`C(atQj1gxAT84qp1uxZx<Y#azAI`VGl?~_n-->}yfbCilsDSd}EJ9oPFua#(_y>LPHUxw|csmz{0yr1W6*!86P=cdK2sMbqkEYt;IQ(d;8;-+|rkde6{Aj8d&cl!5Zho21!`*N5dAR*!KA+!^jT4ZjTHzW;AWe0`^)lScjc~mT`Enmz^9-cO0t~ePR6sG1N}wo6HQ1(x`Qt}?Jinh_6a(`M>P0ayzoDL|AjQ3%-^}OX{v89M0FHrBfoUq3UsR8ykmlafFN%Np5iEv)@Sc92g}65#Y95BTcOPnX9Ly!CiiA`H<HA{ks&V10Kvgj?KZ8{<5Hd2#h2i#<3+MZ9j)A!V%`q@npeY}Qn2_?}tU=B2FE_%q83N`$xHjj)Pypw`Pyxk3DuE&))Sw$D&KlH>5oZPJhJm>iu3fbu`5ElWhvD{>4a5B_22ufBD*_czKAc6UD<94pGz|Z8BV30eVD5wKa4wt+Fq{iR1r!IV1d4=IgT%sa+Tj|fB3-5M<4mNRX1K<QNLRgZoQD+kQbdgdk)qy;sBt1v)N2tnPDDy|_utJgFDqc2h;-KpPs2TEgr^}N^uci+QmQP#Q48b>EX6=5!BP~28pN4MSIuyoh;-EpFU7$81TMwE`~=2nNU84jm-#&0zhfX2z%ejaAWlTOYlY)Pq`OXdEB@t2a2o<rX1}WfLIH+a04m@(2qjP?q#76(&Kk6h3ugs7ih=nVJc@z&89d5`^X(ty!f^kNflvU)K&XK7;VeQ&`Eb^tWB3P+@G%6euyclU;VUoTT)5-~6bEw&+NDTHH84(`cMrLY5$D}QF2lgw3b#wOA$cE-OZhO|zOvzb|3xtn3Q$}tk}FV@59h5!it^#Si%K#4%Z+d=hJd*bZpFDU6u`MKR6uc%N}xyxHHdSOVcOvqry|32!!6E4hT4-aPDF<3g<G75l;&=Jn9sxAPxE=W{bfE6_rJ|AFDqc2hz!*Uw`RCkVFxKgzT5}5oDp#^MzkCfX|e)}fm8xTL8`$tHOwDh;^SHWnqpvn0-IuBegbn!M7<W#a!5qI7ty+7U@ky+41@}FQ^Bl3aUwF@4t8DfFF%6a5D@mT%PEm23vko|xdOv+FqdE`5<(4zap9~%!?<u(prIHDTlpyl=4Wsy7lzwcE)4hY7zhP$420bbl@DhTn#zZ>22I01XoROBAoRi0xiA#KxiC~faga)&NJup>PMkGp8Y9jMv<w5G6<(?h$<N?YKAdlVDI13SR}7>AxK;!zpnN!s&{95}HE0?BK_k2j0ih4x&V_RUwsYZJfvq?QCD@9DRD;CAsoLQ<6`86Vjx&*|n&CJRnW`7g8IkU8{+Q3_yUz&`^;$&B0TK0HM2i!V>00495t*(NK8AbH2p>Z}=!0`cq{{*<wE$E=F_222D41)|<4k0lX1K?R$TYohzZ3)W_R5!HVBTLjr$oBD{b4>2_wN`81#k?63T#usyv1~!h)mT9_oDchcbP7RfVmIuIVIv=jOaNg(q#o42cZOtgj56L!aaZ7xNugWq8ON;!J-(LpTVkJINyF%E}ZYbItJzfRL4N5fb!uiLRI-N?C@{+2ix3NLqOQ)zB(6%0yr1W6=;fsbjpDuA=IE5C(atwj1gxAYKDQi74A*7A^90>%7@|hl?}uFD+W>lTq}|*(3KBo5$ei^vj%m;zuXA-ZU~tB;NG1JLjjx%Lj@EEsRW9IRD;CAW!m8$ry|RA!#&PKmT882oQN!ST3^nHsFxyoj))9*_uYIRZhx51!~IY5%gYKFCn8IA!hIO-<wm#<Lq6z(b4J9y7}0Y`#Jw5Or(z(KU@8hy4Z5je{`e3d&-ynN1M?F&6$9a{rLD@BKYz^U^ZhT!z+8ak7?>*%CnC$W!f_(9TqnE~|MDZa3<04J&MA>03$WAzPyxq5D1jm&)L<JI&Kk6h3ugt|ih=nV+=_wu8QjW+;r5ja!~Hu3LIE5Dp#sW>vj}bF!&!rl;U6@@#}E+u;Nx647vMM-h6*SSQVA3ZsRqW0;ov`G#94ukVIZ`^N3|jO1iVZ6a6bL-QZ@`HA1VgI34oVtMW6!8hw~wTm-69!3gBh<2gd+jhJd*b9+z`rD1dX}T!ErE2qh?rgiwP#7f~+;j64=mZw8D!6;ZDSj64)k?*@!G6RF+Z|1zJ4``_mCaR0}AKHq<yinx~pMjnf}w*y9149xxTsEUFWPJ!*}cL+5YY5}N$Vj<N)(U5AeRT1;&kNJF7!sd9GAH?Q(n4iQr8L8?Wfe}X|wYv&97D5Fa3!ws2l`xA@9*?-U1V(c#gd>?14J(|=e5oSl8pHufb?*s`?s%AM&=nD>2FjDO7<J{zS&O>kVSW_5<6(XlyK-fyfO6$rf#Fz~D=-`ja|MQT<*Y`-xpG#cp;(yv<1rKsbALRBb7iQ3b7iQ3;vv;Q5fN%Il`m&8n#z{57EQ%N=#r;vQ1Y`ll`lgDlr2LA6bq>Wu0??gI9JYUG@UDFHCl>=&?hfNL+F#2bLCuv<y;wRpm<0%P(-8}L|;}f4vaiDQEv{6JT+0T4vaiBQST0nI5Vm0-GLE@CRM#VFyhpts&@xQoSM|(?!T+wUzWl=HgRtcjI9_5eezZmgg!aWP3n*}IBJ1hgQHjoH8_ffP=h=>Q7;gTJUUTt5RBt^m><OBc$lBWI60}qRbZLVLj@cQp#qMDu>vzsPuxocGmlT)TLklRESz}(MZ?@D&p1D+L)KuZ1)v6whfo7WM5=-E<eopTJUMF7ERKgYFW`9C@&d}0a|McWWvGB-AymMz5GvqY`N|78SI%lw6bo~oJd2`X?vrPAuAFO7oh#=WRK-K6K~+Sg8Yo}RVpNqaXDzCVhq+Up)io&jS**&Jp#sX5a|N1WAylBb7A04pIakhV)SN44HEN25xlf)=(J=SPvpH9W8aP*m8Ymu84HOZf26^|bZF=RIci-BkSDtzIt!;YcnK$3srdOW%eD>+?f0)n1{ZI3Gxc_B75BI;-?=MSX-g|4SK6wtsz}zR#p(vR9<eATCckford`SD0HBc<18Ymi44W=q${`@kZ&q_EP5A%aK91ru8m``a}Z&u8FOuKrsVot}xT!HCW2o>n6gjtO89$ee?$<whgKZ#S(5c=eNR{NASIBJ1hgXMUbYp@g%p$1ENau%bdJUMI8ay-nB;&MF9&*D<93>8qW3>9!JgbFwo<_c`*%2|!JbLFf?Td@%O<gI83ee!m$3^j1B3^h<Zq#7t9QVo<ZXEEB!ma`Te#Y5<nk84o!vv`y*=L#HU%TNKuLaKmkQJ@0Om9rWh=gL`)j$$G7$w$#J_sQ#Wu3Ym1&XrqUK=ClwU|oucRD<Zt$Mni8@6L5hue|c^T*vgvD{szqOs~A+45co2|Hphj-+!E;RP|oPiX)V&-m6$SgK|`#ymAQTs6KfW#lYMruc9cJ`{b2#C`;C0sRf`0iiK1IMMJ1T&Y~RCDX$zwIi^!y)$uSth}H2hKZ!YsvRnm*`8-s>u@EZYSO^u^s)SjLauDUHK6y3A!u%vQMZ?@Dubf12?^diFMOm^2j)zbKMMSEB^5iT=O?h(GqULy*AI0W)n4iV2Tsc>uD_71H=#GWC0^P9?D&Sl>t5J8ZoYkl+7Un*Abw$J6C$H{Y8EW8MIoDt)9zqR<A|ljaC|}NEG?Xo8EgFi4xl>-lH7NO69LkrW0?L-50*ZxH0oS7B3QXt9S&gQ1<*Y_iu@L&?sb~m&@^r2YHE^yBHBdaH8Ym)C4I&dS)hp*Q%B6bcJVm)wubhV{m+F;s4n@6Jv2qM$yZi6v^Kk#ed>-z9s^4Fh!aPE`T%WuY1EEh|ih|H5=NyWAw_@cWihH+WZN);U!B#Y+8gx~}{P|%%pOtVs9_9yeJ09jIF(*;BtH3dz&lNb1g}DO9u`pL4Pf#w`C+88$<@)5~SeT#0qi6_ya!#UbS%ak(fEqX+LJbrVa}D;TJh|u3D^DKz^Nxo#FW`9C@&d}0p#sX4p#qMDPyxq6sDN|jD=*+&IjYeviiNpP-bK+c_sP3BSI#vk&Xu7CiicDKMMSEB^5vO7uWUJMQBge1o$@ZOLCMczRlb}nP?as`3RJ~Hs6cfs3RJ+ka#o}2Tsf;zRV>VX@~(=8xli8JxiZwixpJ;SQ#^zkG(|+HL7t-&(<|>hMk%IO-g$~rOs~B25T%%2dFLF;areK>=i&ah`8?eJF`v)(pC>3q^~pPrP>Sl4cUKI|ee&*#g1JxLIfrs&4Tf3(YM@w1HBdC98f;a>{P|-(pOtVp9_9yeI3DIFF(*;fn-x0;QI4yCV<A+)u@EXSRSB~g<q=9zeexcTh51PwiiXf9=Ol`Iw_@igihH+WPshVtgQ<u}HBg?M#b_!|&RR4b5A&lq9S`%fIF&0y1(Yl23M|LMT!H0Sm@BZHD`z!Y&XuzoEyY6Ulb50)^vTP)GStAiGSoovkZPca2sPNsm$MjcWy@KMw&Edl%G)(4`B~h`m!Sg6mZ1WQg;W98qCf?lD`z#@&Xuzo9mPWElaHby^vTD$a<0K~t_(F$Jfs>ZB2o<^6RYZ#^BAS7UO7)us_K>V5T%-4`NSDYQ}0!rI74aby^0fOC{4XrapDN2UGDz7`u)wkz)%aUynte0?vu}@D46@?6K5#xk~KJLfn0;4SO_&JiiS{wJVU9bQ$BfwQcb6PisNB^5R2nseiGvZrCqK9%X}Ux;8+M1a4gIf$P<*R`s9;GC{^{zr#cqqC$TCT=05qv8A`ik4Tf3(YT$SXHBdyP8YoZBVpNqUXDw=uhxt)#j)(bKY|52$1)6eYsDNW3RKT$iD&Sl>t5I{VoYkl)7Un+rG)2SQC!g+IIoF^&SI#x)iic2xu82rAP`;eSs4H8}TGSN}bEkZ|Yf$pD*p)9s1(Yr43Jk?UsK9V7O0K|guAJ3qI9JYUG!zSSpL~X*VeXU9aIOqBaIOqBP&}j>C?Y})@*Jh9UOA6Zn(CGF6s4(NIS)~q>XqXRr4@Jo!+aj@f11z3{V(%*xc{wwe_0Ci2&K6`c_{`$pS%<Wp-+x8l;+;8IB|y3imZWRA=N<9kZLeh5%cGl`FvKw<#?DM#N~LHpTsypY3j|26DKH5y;*U#V_~kqb}WPnbXCGEMtOwNT%Wuh3-gn>6%C<Jjx&^2WDSm5AlKkH9_AVxMMS8<QJ$Q|=qOLlT67!_^P_ki5A(Bllq*98lq*9891EcWj)k!T<+_|LU$BDn<pL`xBIZ{4VFg9S+$%q<;H;VI;JlgYpy-HoP=v%fD1)A`fHLTn7buF6xoLh_!8I!bR!}AlRVd1&sS1jZSOwRzP!*g_S6IRMH0V=Nd~EPs7DdPg&t-L14Rxr_tDz26F_P*~6(z9_k(26$gNFr%S%45svj7NW9<Zo)4jvXreC(I3!7&R!4dRHUU#<pm#L_QUgFIq!FCILs(9{Z`k4<s0f<87y#tQlv2QB@wby#Wzs)OPq)<F@H>X3&m>IH;{1&Ub!)Ui88!eiMTBjLG><CcEODh#s#Rl)I*s^IuYRoJSIAWwPd;@(6k*KmA<=W-}QR(LMs;H6)(4sr6*FIxx4NUDRPB-TMWHptUZjt%NG93$bu9FCFjWKQMTP=%>H8>%oJAE64<@sX<FJR9_BI?o1unu?FqV^2j$>anNuY^sCvY^cLhjHEg&MM<i|Qmzg1w3KOsIxWRW>av$>Y2e9R%C)Hq%CxBpijP<Y*VI52w)1Szr|mo&^l2+TQjfhAA*sjS&a<fw&a<fwiji0cMM<nfbnWK)>^z`puFuW`n&$fKJfLZ=&yE9{UR(|0gr*l+gKid}8Vs`l)nKY$BgkVO(lps)AH_xLv5z7n^w=fNX}WuP;bDcMR#>ot;v>|dT#Arbhps9K3k<VBg9RKTFId4bvcL*k)e#mrW&x-|aeRa-6vs!XLY~sN7a1N_XljK6D>y#Fb6FH2p~o(9PScC6!%{0y9ULR64vLaghpHSK<f$sh26d{Ak?>$v$4GcGtMY8Bg7R#tg5x7q!SRu*;5-}jsXEUFeQJu2&|{aT2njuQY0k5u4$XNs)j=^5>!2u!bx^Ji^3;@RgE}?ENa(Ulb1e-#nO(UyRG}-=hAMQ$N2)@1O$}ASc{b=%cb*OU)D<70$1YtF5_;^?oo7=WoM%HFhGHbuVJJ#c9pbE}b=7B=II6i>fDm-E00?9p)Z8op`WWXly~-LavjEj#n+2!_$1DIfh*O%|Un9t49Mjyc0Qxu;7pcddij34_$2m=}whlwBKy^@j#5yQKVjZ@sBrI^u0-%n|F%llj<roRiWt`G<_jX5#Q<`376&xR_3XYFdg{kTY@)XB3w<~}?EyqWAE|(%C_1JMv)8+deCC+KOe7~b?$4IEdR+Pj#D8~kQ+RCv(owj2nJeb=t5}wShJe#VZJR7QT93P<y$MF%WaGYm@J{{-Tpif8fk$UW-2uVHmah^?eaGp(dP>jSnC`v*d>ZM$J!UD>)7c8I{30-!*TuU=x1?AdQ1!dY)1;t0Kf@^B13eK}Dtl&I5>r*X?kI-Y+q6i5+b}i1cp$^4)Hq}8f66>HSiFJssJyf4v^Il~`_1QIVRyI_hUGrvTL-pA;&S|=Px1+{6O?U5h)HtW<?%j?W$27gk8g%t*1bNJ1n&EovS``<e$F5b85qj*J^EA!Y;iwg$4o&fq>d+J+sSY_)GgOydbEIadF1t3zNO&xpV<bG6IZ@MO6_#0ms^IuYRd9TSD&#!Pa6NX-ftulZ?Aje4;koRJkkDh-oTzEG4nwU#b#RQNIw(qF9h75(Jay&RpiaXv5+2Oq7zt12P@WA{7|OG$3XYFd1;<CKg7a+9r{O#s^l2zQLXTaCA|&<L(|I=3VLH!-I!wh#s>4*2#5yR~26>vwv_YMwVkC9h)3r44WKQMUR0U<)P=%%VNL5&_sevji=h>i7%Xv2F(^7n-9(yT5QjfiyXHy-VXHy*%Be4#Ol2nJBshO_N&XJnw`s|#jnXb>yftu<1?3}0RvIfH}KsA_V0jj|=3s4QV`Za<)<}l5aJ@!#tq#pYyGE$G7^EC4Pj+z5C-PS?z5$m7`iFKH&lCZ!s3xGNv$4GcAk7HzhE?Z92xVJl64%E1}JKE*=c)<#ek5q-O>bPNrp;kDsg5%?a6%-+%$8I@M(`_A&S^?@%93!C)MNyLKP?TeLSU@@UfCU^Q;lV79k?>>|<=IpP<=IpP$49Dy<0Dj|I?o1us?M`PpQ_>`^w_N`LPC$-s`G5BgY#^vgJLAsK~WOxpj;c|sVdV3b!v){&}FygS{isVn{sWaLQ|$qRZx7yD!8VGs^B~u^r<<|27PLZkI-YcrU(f=cI(cwp$^@7Hq@ajMp7NRq9oQKy7p3icFU2PrTXla6E#cq*)0cZmg=)x&eIH8gJTwe8ss#Md$*(IFpYb+qvb5kay@p-VVdQ7>^2k^p~r4Rkr8_Amh&{j)?uj?s1AycSO-N&szc7yEZ1e{NX>Fx_H>Md$8tJG!gHAuHA7Zmm<6Z`j*nCY$49EdR&@k<%7L0Cd+g=-2+!qGgrpujCu-#T9W6&{hOL8RB-KGt66>HG8{}yz#|Cv;j*;+SF2_iCGPm+<sKQpB4OQ5Vk5Gl}_()Z7o(=l6oo9nSZN*3Gv9}^5_1N2aHr2s-Hq_xLMp7M)q9oPfDAxvgI?A*`osME8b=k+YH1K2|<=RvQW!h8)#Ye1yYie>8`sF;k!V1o_8?2!C2t9Vc6d|F<?w9jys)O@vs)J%A)<IDc>kwUgt3JEuNX=G#cF&2Lt@`Yq12tRq**)iJ+`An;$7!alK{pFf4Tf2OYB1HWal#5qt*~GP#YO0`dr@SB9=qo}jeNhO=Rl2ozoS>hN2)_rgv2^@RY_Q2m<2!`t79ZQmeny5p39u5nX(GUEC5w#j*n1<=J*Iz$a$LWdhDJ9HQV*ry*WO@bJ-Ljp~vnyQ8R5FmRf=8;224DP?V%PbmiC}PhB}Ss8e^0ga@-bM#7WXm1k2GlxI^F93QC)j*nCY=h>i7-FY_X(@=bb9=i`kNa(TqaGniy7|ye)4vLXj2SrJ&gK}+<r=d(6)M+S2LYLi#YiZ!goXWMK3R9UjRADMUQWd6aYN!g%vq7Jx^K8(ksrX1e_EdzV9(y{^raCy!hB_?8NUFn9l%zW3OwDn9c8=5>*JtNM&2fEp4%8giXXiZ4k~LUn0jj|^3s4P?SpaH~vouHc*f~sdWRJZS7pceIij34_=RD1_br@;|s)OPq)<F>x>#$WNVS!^70ChZ$k?>d^$4GcCbE3w*-O+QPX2~izK2jANAE^pc)e+<=2WpP&v5(^;JeNli5_;^B6E*Vvj*%lZ^8Jo+IYv%cK~WOxpd7oy0?M%mEZ`VfVFky?1}iAfrYb1UhAI@tN2o$^e1s|#=h+vm;5@s)3W|@=V~?T;2|e~G&a<fw&a<fwiji0cMM<hdRjv*4RF!FiI#tC;=(0z3Ee$-GRk=1*L76sHLGcl*;F=n$g7a+9r|LW#^r<O6LXSO~A|&+KqdCupIyC3mR0qXKtb?K?)**7zz1J}EsK&k5F!H3vz1J}EpvJw|FyfqMxc3@HoYM^VUc-oUn&IAS7;#K9rmR6%zebS9Jf@NFHH@yf2tD@bij2@>k2t3p)7Igr6`&47@sa8<6d|b&c~0ZrYZ!S<<KAl+!!Z&b%i$Oa&t;s_j47+I%mP#e$49Dy<0DidPif?P4I__f<a-TcIzGa4ITazP$BuKFF>M`&T7l}|7)f<dl*BqH#|C+t%CSM6mSZG5n9DH|p3J2@8>+CBXHyj%AE^qCk5mQc*`QC$c{b?NQhcNydnrOvkG-8|LmjsBY^cLljHEhjMM<oKa&3^OtxOx#X)8ujm%Uv}15f5wu1!@?rVUj%ijP!<<C+?%!f~Dr`gELUgFYR_N9wVUA|&<L$9Xo@!Fe{-K`|2RpePA-n0Z#?-fNh7RO8-jn0Zp;-fNh7P~+Zfm~l=smaM@r3s4QFS%7M=%mP${t$vLYRyb;ftdFxOE<%qziy|ZR*fY*)hJ3GK#yQPcwhoGqSO-N&tix25gawvapuqx;k?>d+$4GcC<CJE&_ZnuL(hT=r!>o>vP=)IFNLA>njv!BYOe5cGnAPzSp3ABT2|f0VbDFVi9gbQ7>d+h`p$<(^lIqZuV}m?3<=CK3%`p-l%;p#gPi9k|O;u2yO;vDwq$)T*LKV96Y|y9fJR9_>D?UPxJ-Z?#^w_gI&!##!&!##kMq(WlC9w|5wLzY`GHp<&p%@8W_8hLIfhTh)*M=$#W!h8)#Ye1yYig(p&a*+EhVyLDr=j==J@y=mkkn&O=h;w)={y_iFcl-I4pUJQ>kv74xjs9OYA)Aj=Sj`w`s_TYxm=$e=QLx>8XU6#)F94jhI_YT#xc!s?{>^QrMYB}oyRnn?6H^PBK6oyk&$}rIHwug)?uj?s1AycSO-N&szaXBT&~N`W17o#+1oJ^9?R_*3D0Gm(u^&uFw6o}1;<CKg5x7qVXHcVJmoRXC421S_z2JCQG}!(JI-l_e7|GHInCI%4vvvj2SrJ&gK}(;r=uJj)af`z!h?AnBlDBFF6G%B7Eqo&U;)R+3M)81QWczMU$BDn>;fw&K0=SZE=5S_vDf81o9f^^8|qLLBdHEWQIhIVlxt5|K$-S}1r#Hp%U;E`H1K2=<=RvQW!h8)#Ye1yYighh)p<7PQ+1vV`cxGkp~qfT5fXarRh?&39h_%V9TX$64vLakhsepI`s|fQHAVH=D^F^Q>a$lK)D+ccuQ;a}?%j?R=QQKU8g#P&)nJ$fs0LI08bKcOn5MWMdo{&H=&@H*WP~1j<vfjizhmV<jeNgjb;U=jLsx{vI&@V@SYVh1KpnedBs`YgF%q83oTxdn3dbw}RTz$sP=(?62vx{an&NuwmB%#2_1J4TKEiW36d|F<UO7>7Y#o+bf$HEGNp(<^q&iIH*dR|+IX0-%bc}=tb2>)ClR1@VQx%kFQxzN^sS1vdR0ZeRpik3zHt5q*e54+GDMC_@y_{!59hUQKs)J%A)<IDc>!4g4<Y_6>26bABk<?``*V4d~xs_`}6}B>MsKQo!q$+IJ)KC?iXM;X%=h>i7Tk(;4?5zk%J@$5<O?7ad4Rtt*kyMAHC`on5vzqGq>^!QeuFuYsn(F%OJgBLz&yI7NdC3|qvjEj#n+2!_$1IR*u=A9rx*mJyF->(n_P!Jsp~v2rA|v$JJI-n5W$Q513RDNhN34S)B-UZ8O2PuiERfZ47stp8R&b0gutHUJgaw*efU4m5NL6rrq$*5R#|<kiwF2l<aeRd5vM54AkG<oZX3F<FcAV2p`F_W)j*(D@swjzdP>v1qRFz|cI#tI=crdGDBs`f_c{Wu+c{Ws`IX*%an&Tr>p*hb6eQM6LL7$r9BlOt2DMCVzy_@rFs)O@vs)J%A)<IE{>d=*IgFJO*+MrHdF%r7$-CauqPi9xHO;u2)O;u2Q#45O^hN|E^8}z9=&jx)OijUA^@1Y0@J@y{Xv!M>dc{bHSF%s*bD2a85uH96hz4NH1sXlw>NljCI_RfQv=KAb7r<v~EjvePT)4kiV<D6!?cRO|*)661k(ABRI<S~zFn(VQs;v)6fQ<0H+>^P^H#n$1d6`&4F@saAV6d|b&c}~+@mz~En&2`z!F%llj<roRiWt`H?BCD{>0#pUZN2-G3BUB+zX`1Y@^O&Z|9(y}J!gIM5A*siXbDCLf9fn$g>fjhjbx@SVIw;2mdD_acL7k3cBs`eMF%q84qdXg`aFl0L6&xR_3XYFd1?SnIPse#S=+jYrq#pYyLPC#yF6Y@7tl&JmzzT|yP=|9VN@5+9Yfo4}nf8JO6eFR_K9_502CSf5o2sBp8>&zgAE^q(H8oI$;yk;;3eK||tf2S^J@zSzkkDhF;yjz`;5?h^pcsjDP?V%P<V;OhefG(bny&inlM^*v_1Px}YP#yPPn^@tDr+#z0#t)(7N8m|vjEj#t6w9?V-C}F*JGcixClM=X^M={W1l#unezRP6X!Iu+BzscVjUDAu?|yJ5*Aoy0Z_;07zvMMbBu)NGEQlxd%NSrDa~|mcbx9{2vz8gk5q-O>Im|b12x_C*rz)_!gJXbA)&`UaZWR<t;10(Kplo-B-CLjN>Uw$a%_;Np&T33X*fp0gE<@{;mI7zv#AQov#AP>k5mQ6N2tPdo(=jmoo9nSO~ps*v8N&=_1M#SHr2s-Hq}8f66>HSiFHt}4e~UVX@fc~#YpP1muqR@$y~~{p$bcxHdR6K5v$;u8mfZxY|y9WJR9_BDLzt<y%Zs-$KKAfp$^-5Hq>D&Mp7NNq9oQKy7q8=c8=5x*JtNM&2W8o4%7_SXU93sY_bN&EC4l#bDHVi?Kp8vGu^u#CueDf?6Gs0X2>4<C@xZueH0m~$If}0X6vxj3RDNhN34S)WUfQKa;9dOF8g7DVistyfMetZD>z0LSfQ&r!UDrAKvi&jq$)T*QWdtU<AxQETH)jsisR#i6%-+JkNvPhRh5J~<Va1kb#RQNIw(qF9h75tSU@>8s8ex_ga@-YM#7U>m1jd0s`6~8LUnwEDpbcus)F-u(5LD=8}z9vK2~@xt0H8D=dwD_raCy!hB`FGNUB3ql%zT|<=P-mO_?^RQ&WsA@L)FA(!i71lxtHJlxb5H6d$n)uBm}4bm!TiPu+Po=u=mGZ17xmMaTxvWp|!Ub#R_dbx@4NIw(qhw+?Uq<KN%@i(h^I)zfdDynONc)w3rrKL6&~%RfDR_0@~#-#o0_uQT8IH?N<6^~cxGzWt})^W$$`y?FWbH_x8@_UZ5c_mfvIp8WB55C7)h{){w{&h3XFKAeBk;lt{Li6%ZAuiORfDEmWvI4b@Wu;Z1N_;9@P7O>+Zb;O7BH+Pr7oxeFO;?w!s)i5;kyP1#YZ>qXsDIA<@J`Sf82=K52k@z&cMJ%AhH!|YW;H2i4mhD4{TJ{bjYjAZXZ29U&+VUS#M%(gVQAXVIwT-&vYa4lkKi`bN<-gsUQMi2dW+V=-o*1WMhOvQA8N=8;sI8l0=b-j&j$MNaH^!DhB^zVEpf-$-y@J{;I`#=FJQjNdl^l!xf!dHa_6BM<-q;to$XaX)TwX191a2eX*bcZ|fa7~xDijpo*;0w1_`a6fXf?j8rFK~zM%%KR4kd1pal@z^TwTdqzPi!3=%>TzTlCXm^etc8=v%z@H2Rj^bQ*!nZaR&^!PSk#!POJv)NB0l{VVkje|-0<sP6*ei&sT`8xY^RQt^TK%9Y9v#P_X=`bHqWY*o~E0`W~N^@4VM%}TwU9pAAk>U)9sf>lx948*sq+>7Dy)hhQ^czmy_xGx9dTUEt<I}qQgQZL`e_o>vIxA9%7qP`r6FH#lt?Z7rFmwk0BnTxF3O6TD6PU!OWjnYML-A3u6w;rW*@#04*UB0?ey6mmTNL}{UqqGh#ufz_npLnNUn~U#Bsdwh$J5oh`Nf2L<DsCsD_<EFkUoXBK<=)nduSONO4N`nDs<>T{;%iZE&ztyCl-uzpz7kd3W;5}HsN(jTiLXP+{dVHZP;$4OIDRN{tD881C~~KpID~Nfn#3W5+tnlvA&T2jBu*iU+fSqz*@J{nQNow`SCsS-04IJhgEH!uUy-dw{qp;<RjD6bUa22k->4s?II599$Z%Ape*{1YAR$N$bo&#;;e^|rAPy&r+ZrHFCyLt{AdV;8-Ux9%;dVxd1B&9dC5R)6;&vs7LkgEck5dYlK#yaJ;=<o?P*Gg=JB})3{yh#WWcodhD~gPs$B9Le$@4h0a5>{Rv~VfoIJ77(;2WnF#pQduQ9iKVuB0#XZz$~p0K<tNATa8e&372}%icSb`oZOu`oZ;$`hoQhBY)t$L+Kv@Pyz@Da=2WXZ%&pg?~TKZ;-Y$Sno(R*FK_>=%s21-tGqW4G~Cw?d7|OIcgR6<WxhE^?#`~g40<^AteDT!`RazcbgZ}eJe?k_xJyU(ss#YS6n8)eA&QUF`F{pJT>hBP)45XS_eTdRna|UyE$Y`tN2aI+0D)0I*izT1AMJZ?;KSvW`q6&A=J!V%*_zMOZkg)WM_W~@1pq+|oVlEQ6CbBNuM&K@E#ferr@dSRe8}i8@p0Ovz@Vos0?g+j*>wr{6u1;0r<|cd50`J|^OT=4)P?lSFrSAE(Nx?8081?Z2)4Kbh?5=h@%#pSo_VOR9cmtVba#EDeDn&wlD^EpqO^|yIPn7nM*Z^3@WrTKejC0h^@GbR^~=|<M*ZM*{A%P6vXoWn9|2GTNC*-GmoeWOhaO8M>+;NFxXUl|dAR--@N=*_PCS<K-aPS8Upv%1@fhy<RltV?P!k`|%9AG^ZY!ER@o+oQ#EHjp)*DA2%Q<hJdB}}$^3X%>gA=D7OPOyRdMxF=dFJ7^8_6RNx7$cJ$_LilmGnUZPH7(jaN-9DjQVBs?MD5w_YS3gkO5Tc2iG_12i7}`{DJchrGErK2_PX*!Q1RV!$fa;_Mf4`w{U$EzlG~h6Tlsrfzu>#2joyq1edOFGPrbo6T%(YfToGzj_g0v1aay5CW=efH(}h7`8Q1(cVzx86UT+?FO$cm>zhpnMmC^jb{!bmf0o&HfUa-$9iZzQ{hOJ8%jn<C{98u<==w(g;QCwXUpAnv^e_9*R{96mSNaFnSDO%I{%xawGxKk&jR@fSM*rygDy*CJXPdY#`0r7H-F*FHBD-{b6WRs;Jtnma{(DSpm#%MeyL5dM+y(zVCb|p$drWwju5aSI`TFfL0p9Gt?J^18?7!_Y5nj5!$?($k%|->;f7@ktD#-rZF0)kuUEl0gK-V|=m;JY0M*p(^wqo=Tu3wD)(e;)7!FDi8|MGq?Md=@0U+EuQUu|5F`B#koW&dqOZC(J^H~I(HuPWG^^{1L>Z}#6-UAPwjsCX|SFah7}z^$5uZ+75TRm7J7sEjWmFd^UU!L6E@FL-cML0<sSRMeLcn6NLna5HINaN(xnz63z!eF=ft-T-{Knf(pGhnw2qfB>i+4gf(nN(fHejS_+rccp{`Kq(<1P)eA+xGN>hUfi7$0syCkfIw|`kV)8$5`r5Kx8*?wFq9G!0+s*GLNrVO7~FWc1TX+l5nw`KGQeQR!(@QLjfctr69AO~1_aY&fWeNZ2?2u}Pn7~D04fGd2uuzb?0A|SFu3tlIbZ^ya=?VZ?1TVzJk3rB;Kobsga81R+6jRW7$pQdUPcMQjh9kF0-%(T5GW<gcD$4lW;b3=2?2mp!i-?6-4HSfw^2fH<L&lC002q}34zK4XCc}q6bx>>T`Cv=s8}!|Fu7o`<85-m;KoPgf&su$xnM$Ia=~E7#{`4HjgLwO695$rCIluM40e1>HW=LasBAC+P}$%C`#7835wack%j}Mj-MC+BcLV~Uc1IutMhUYW_sb|@cH@32B_se!2?>Ex!o0KdrIc{w-N2m^W&p)0As|pYB*1>=MhUYU_u}?Q002q}34zK9@BDca6wYqki%SXv02LJm1l44PvmN(pvclPodsSIs0-&<OgurBlvmN(p!ot~&dsS&+0-)l;guvv5vmN(p^1|T8P346FKvQ{PLSS}G06T7G#{_WWrglsq0BXksLSU2-?6?^v1UGI<2?>BwLO{@!5@tK@N(r+Ycc+8^z$qahP`f5%5_Y46;KtqUn*acm5)uNH9nM13O?VjGc)0X%1~627m=Ku!Fxc@h`C)M5q4L87K;?%Cfyobp9S;*A1~(omK}-NtgcuM^lOYB>o+d*KZah_nm;k5@F(EL!D1aSLvx@?_@l?Ag5CFA{0wFL;2zES;5`r5qrGx-rDJ3KXN(r+aFQtUpjh9nG0N|7m5U8CLG6|PaLU7~d_EG=<N(lkMR(awqMB4<4!Hu^|6axSiDJBFaQw(;zO{N&!c&kh?0Z^G@LSQn*V8`2piouPKN)-cuqhiH`z~qX-j*rO|gBu@}D<%LcS4;@Z&I(}1$Ly>CZhX|v3IssytdJ3m%P3*C<8c`!%x*j`rGx}PDIp<HN|;Y$yp$5cS&&W%0f19NK%jP4$cHjsMhUYUkK*=M$N-8`LPDSpHP5Frx>L>b*^us7bGW`b*Bq{I^2HTSBsKZsni;4LH>c~Hd@)?#9dN$$=hX@4Cx2cYaSqp4XPm?JO}@Bh{;6ZmYv!Li=$x)^^2Kz0v%f;kCZtX~ui1mtap!P-b>2B$-{@b>{A))4YUW=v`bXC{`bXDS`Ui(1EB(v<)0F<f^_Bj?smw<ITIOFj`qwi5y3s$nzR^FrzB&j!>yJALJ?oD<3LUPm&O(Rln|v|&uR0DL{8t@_4%b&lqSN)&q3BtE+^OhUf84R?aD8<yI$Yo6i@|@@(dgj6>Tq<rzR4HE_0<9C;J@mGbnstwL^@sH?5{xAH~I(uJB|K<|4yTSbbX_LbbY0N*?*_fzwEzD>0iG7Qu+tiH~I(uyNv#U|1P6{bbX_LbbWPfde$FzZhF=qd2l)aa3`l{1nTH?aA0+KIykU9KAixlTrnX~N2q5Fa%ZS#4U&hb0|0l5Iv_B)VsK$~kUF@qJ4zh@xWm*5f!SdJd{~{Q4nFJ-R3`vxhXq1lln|Ww7$pQJK1vA*fKtMYU|vcIYyP}a!j?bpln?+oB?JUU39}o|%P3)X<9R71BmhbY34uCcedf=r6V{^+O?kq41|UyZ2LvWpob7n3Bi6GUPj|*T0Z_SOLZD7r-}&?El=ZAb@|1M|;7(c32-GR-*^Z|=W<9&{bmy!S0F^5y1ZIbYY{yfbw4U8~x|7xkfZAb!5Ev!Qc08+5LU7}zln?+krG$h)DPgwbrj#(dadS!t0Gtv60;7ar$IU1qxN%cTNC1=)0s?pHdKMyg>U!28dFna<aHp;V0+TBSJ66Z8gB!bZ*9m~i6%zt=@_H5`ck+7HA$jt81|Uyf2LvWp40f!JUI#aJXRi|gl`AF$W`_l^V|DsExUoBZodBpE768FCN(gp5jS_+zPo;zeKq(<1P)eBXcq%2#Zake50syCkfWRmr*zq(<2yVQT5(0pwl#mdpli0Hmxs%wl4#|_)0f0M+9T1pYG1##>iXGh8oyAT7RIV5hxYO9P5V_OXvkuAA*a3h$jU5n}Trt?OI*uLO*qz5t093A+5SSenz>d|4?BK@kM0Nn+PGlzpMhU@=k5NK!<D-<204OCS1WF0B9UrBH*^Q4=LIB{DFd|r&QNnD;>oQ81-FRI}2?>BwLPDTUXP^1=>U8#%KkrUw2LSGLc0gcq#o3ORI-Whd@p9+00|0kGJ0VafwD0_Rbwc~epLZv;0|0kIJ0LK*;%vuD9nqfMc)2s$34qEK0|IqQd$!}HPHE3>yxb}61VHVuKnRQyW;<ThC}DQvRh1GF0HuV4Kq+C(pI1tl-MBd=%mA8GLO@`Y5bU@aB?LEaN(l*oQbIzYPHfLY<W6kQIwVhQ2LSHG_KZNC*ba8Aj%)`vc4xK|0F^5y1nSiGEJW_q_N+tl)OG;iPHhJSCRYr0td4C5H+JW?0|0k$J0UPTEPx%WliR_K-O23)K<%(V2#gYf9S@^~;KoBKApuZI2neQ9!feM=DPeZw>68!vI3)xGMhU@=r%^(1<EfO804OCS1SVhH;8$Ss#V!8^)cNgjeRY02UEk!3d;SHe^V@r_zdFC2u5a?ibbXUAZuyT9b$)xxe~YN|+v)lyUkuk*=ePIFKXrb4&-_#8x6}1azL>6W_E%{64-s{Kd&_@^sPo(D`euIxy1voBp82<p{`JhiWAqQMe~kXo^^N|u{71+!`q%Q`Ajjw*UEk;*UEk<m&-^<^|9a-%G5SZ>H~N>azt#Ee*?+e>zdif!R_C|V^-aE*u5a?i*?+e>zdif!R_C|V^-aE*u5a?i*?+e>zdif!R_C|F_0{?9bbXUA&i=dA`R&<%w>rO_u5a?ibbYhGLiXRS&Tr5DyVd#abbYhG0$jfu{mcHltI@ygzq=a!qw5>}qw5>}%l^Bo(ZB4!yBhtY>l^)}>l^*c{=2KuKk(mX^bfAzjQ-K}O|BUHSDoAr{_9R|Cjcr}ObAS_7#vug+zt-xPHravDpw2$)XDAO!Rq98@L+dxI{{F+VnSeY#o)r~<aTgjcXB%cP`P44V0KslA66%~gAcot+W~+(xt$OgB?KoPMhU@*hf+cUpp=jh7$pQR9!3eli-%G|0-%%-5KN<l;KtJ^A-M5WN=N{d5)uNFD+W7OC%1zeyOY}qfXWpU0+TBSJ60#RgB!b(+W~+(xt$Q0Trt?OI=LO(*qz)?093A+5SUys*s(gf9o*QR+)e;gt{4!gliR_L)yeJP#_r^H0-$zSAOuDU!H&04LU7})l#l=@B_sq!3BiuHQ9^Lzqm&Q;9HoSWz$hWu@i9sWZhVvy5&)%yguvvA!H(6*?cm1l<aPp}a>W_JQ75-&J3i{<_Uy*To!m|URIZp1m|StT<D*V)&u)C&$?XI{<%$V`$rWciKI-K5?8e8P+ztTT$?b%|?68pS_^6ZHvl}0GaytP~J1h_aqlDRxPccfE-S`xxgakk-At0zm39}uaYLqa$@u^A)34l^ULSU3I+wrMJ39}oYs+5obC?zBWCRd#8_^6ZH!HwO??Et`?+)fBgt{CiCo!ky?>`rbc04i5Z2u!XR>{y-L4sPsDZYKaLR}2W$$?agr>g0BCV|Q{p0Z_SOLSS}S06SJEw}TtIliLY^+F^kZ7$pQd?nVj0jfYY~05Fsi5(1-yV8_EKA-M5SN=N{d5)uNVgkZ<RC?UA<P)bMuloA30b#gn{u{yaO+}NGmP5@M{m=Ks;G1##>xgFfto!m|URIZp1m|QW~u{yaO+}NGm4glQA?S#POiouT6$?f3A?&Nj?pmN28!0fO9cC1cr2RC*nw-W%h!vY}KMhU@=w^2fH<E@mC04OCS1V#zLj<-=laO16%kN_wpBm_nY!H&04LU7}wln?+MrG$jQ<ch(L)yeJP#_r^H0-$omguvvA!H(6*?cm1l<aPp}a>W@zxzy3^*^eI<Fv;TV#}5mrXfZ-i*<!?C!o}H>9~Lm_;_S%}3#fQ8LQwf)#9+2s$iDorfZ1yy`*OM5A?^gho#Kudj4XDbV==nefsI88BSKKhh!~7CW`BNIz-VLk=Z6K9I3fh4j)b8ab?iXLYSgg<8>>=BgrL+BF~m{IyUE2178qs$aH%*<8B-!yW&uL5%`E^K<22=F0pMaBr`+5EAc*6Xn*|6#HMam373V283xGq#fy&J-5Q1SAAOurU8K@XXDz_^D7voUnmdu3Vs1*Q1KFs7M%?t|^vj8|$9IV{j10iT;0Yad3u>c*r6Gq@;SIUSOoHP;!rHx=wU1=jY)KJ<82!_%|LZGy<03C-C#{zsDN*xh{l1Ic4_3=2T&ZnCk=hSh~GL}Se%mRQQA8&G;Q|I$dj&tfbYPq=uLeR_tgrJ*S09nc>oE+!WaoBQm3xr^q1qi`bR0b;MLr#wL={Rt?B{N`%6PMc+2tze@0kRZlE;kE+D#fA8%{>r;VHO|+N*BqemlH<d<5J3q7@RZ$hOM*_oO&y51gG9g8wr8ZMna&pk$ie9aU`GKN*xh{l1Ic4`FJ^}&L^K-&Z*-NX2@?g;t*!YZ#MGLCzo^TeD=xZoH~wShWutDj$wxUW+NYdayh5Yr=MKTspBAK$Zs~{AZEC4HcCGJ<Z?b;@&PE9^XU>NF=JZ{O|3u}y19!E3k<WsfCZ+x2STvS0)#;6BKUN<oG=0(i&93yP@FUp2BnP?7Es!_U;(9#gg|K{AyC=~K3$3uNAT%VlsX~?C69z5PGUw;PF><8=4Js<rZ|ZiM<Qru0YcEtEo@j|m<0|jFwHFxf@KyU1l!yK$Wl(V6z9}=4&%PrD0vQZB!a4_3{=d)mLh#RPhyV6Fw_c!VVb)DS<2~_;+#59Vva;`%mRR*D_ta??oJqik6kGvVsO$(7?d`GQ+K6};M84dBOy@QNC=cRl23Ofj^xursUu++N*)nI<YRSCozpJWIdz=G%u6B|W&uJl%`Jda=e$dGP95hk^O6XTSpX2kIn2#1fGov1%*_IzN^uS|FNvU;1qeY`R0b->LCoz6z{NO;nfAMlIEtC}yNx)Dxw#9Fr8tVYSpZZiPGY9~W+M(_ru=4ODP1I=UQQT+k4q^dVsO$(7?d`GQ!k~B;M7ZLBOy@Q2ne>)M)K*c#F2b@D|JK+N*)nI)W_zWI-ft^oKwe1%#`13#6irI-)!VLOmj}1=P=DVb)3UY`OQY0!%X?jMxMho=hS%))0|VsIn0#bY{WUtl;3RRIZTs2od+>Z`gEMcO#9tN9K}re-A2u`n5LY%=21*jPF>?9W)_K{nFR=e(nav;dO2Ziu!2%X#NecnFeq)Duz=FW?9{a=Z3F~GX(J&}+6X>fixNlh=~|RJA_gUoh#~4@S595?D5fi?u5l7Gt3+_j0)QaTVY+han&&WGIdzS5m{}!)W)>g>-P{7mQl7(f<<vFKVP=&GmRW!hY(-_DVjjeF>C<r%Gv#+1HI8CtwHT_o3y`Heis{a&<0NKQiC~xo2!YZ?^6BP;5%}1YG9m^ijewymZ3L(8N*lqcyV6ENptO+?C~YL4?n)fVr@K-|#GvF6F+_bF&Z+Y#W;myglbGqg*{E?4Gn+(E%`Jda=Q+%9PMzm4O(Ga(0YWg%Er2ZLIm~cQo#!x3A~<FNK#=D!L;7?c#0=@vc@onshNe~^4BgxX$Wk7~4CmB&5+lFa$b%U9&Bj!^NIpHCFajT!QbxkCoHP;!rH$a!OKBrG^-|hM2$VJw0;P@Q(@TjX`Seoih!~VS5{B)v)L>Ivg{eFE)OKlV!l2^Rh(YD42T*dWJoNxhZk4A-3@T5J7*w8mfLCo5s2<=|M<uEW!%>lH#Go?O12}n9rg{J;kIGad29>Es3~JAb0e*GVo)ZK7>Zm;@5QExt0x+~o>Eyg%1*Ma-zzRwy5rfi6#Nc$Y!wOC(2dv<9k}x=(Bn)ofi3%$yot(hQ;&z?@3`OZAVsQED4J)XCHF$R`E@4d=RKyxFsEjpucPlDm4c^^~%2*=?m9ZuaRb{NfyIWNuYw+$?RmvJMsF*ckP&sSx?p9UK8oawzm9s_+Drb!t)IJp8-L0y9D9F2;+J}NLG_?-}Vo*9s-rbZ=l6N<ylZZj-Bw}zn3EthDPJ(wgr;~)i=_Fw2ZZ8VpWLG*#-re196of(PBw}!xYtX2!Lf7Qo-KDMxgNj`v29>)e@9rvhP2N3J?wT+RmAgg^DtArZJyh_TynCqRHDXZFYs8?k*W}$pWv|J*hss_f29>=g3{&k*LEb&p-W25BQ|(QG7}VYrh(YNjdG}O0N!~q`P9g@SlZe6TBzX69ItkvroK6CU<#duTxcw=BlS}C&dG~TVR1gNGlZe6PutB4i3SyIYFPFq73@VCE7`Dn{lXq{G#U}6GDvOO6R2CaCs4O;l_f}zS^6stD*oZ;Ju@QsHW0QApmB%LUJ}Qq*7>>$gBL=l!1$p;T`&E#4AGKcvVo>{4AO@wA<lRT<BzgByI*AyRP6CF0Ii2jVg44+XD>$7b3{EErgWI#B!U{?!!MpqAcC8=`N+%J6%Vuv_L4~uyyL)lzY`{=dJR32nd^UJ@FDjo6-rbAJXCnrc&qfR?pAFvKiwbCiclV+a+K53#v<X918Ex?HUR6dLyt`MG(MAj^qm3BU9v0x;y{bJdz`J`@dsrX_wTA^_P&x_T-K)|`^6sW|k}x!-lZe6TBzSjoItkw0oK6x3r;~)i?PCF)Y)U7|yPMm|f-oqZBn;i<wLzo03T%^icbC{E3@WmX7*uAPyt}K+HhFhfnQg?NGTVqjWwyz?y9#ZScMp}?CJaNxwh@EMZIgEomD?uo9xAtu7*uW>F{phl$h(Ky*MhuzsC_LEgWA`EFifSB<lR&0BzgB#I*AyRP9g@Uli=Oc=_GjfbUH~GoK6x3x3>jwaw?r9?_O?q3&60HP9g@E;RcObD#T6Ry<Cc$FsK+eVo*75^6sT_+~nO$<+u@p%5f8ht#aJt-CG5@$-B2oaw7&6<wguD%T3<BRhFB)d#fxrVo+Ib#Gv-NAn)F4uM6_-qxQNW3`gyCff$rdl6N1aljPk;=_F!MI*AyZPJ(wIr<35_$LS<ta5|YWjLYqJQDFt8li=OsaywoS2BnjT!R5L)te}G3;N9bL$!@}+qTPr=WxK(<$EC8};N7FBY&T&jD%*`1RJI$udlVJ!2Jaq4rMnS>igzOhmG1`c9!2H5!MjIM`EJCZ^4)}?s{Jp(yGK>~Ux0Uys`kG?3~K)i#GrH%yn9roli=N>DxE|ON+%J6)5#7iIGqIVZcZlwLvuPw7~CEhz{#d`lDxaQT`&lP(n-Xi^4$eItEqf<0nchG-;EelzMC*~mG7>=$*%I<6*$>dz8f*9d^ciH`R)Rq)m6T`fM<1;??wzN-;EelzPkb^yUKT0;N(#GZo)8Bz8f*9eK0C`)=>LkRPd~!_Q605Y99>5pmeeVCx_C>3Y;8DClQ0vNy0FlPJ(Amr<34W)9EB(a5_mClukC_<WxG@fRj_{Bw|oHi5OJAo4k9fd^dUbQu%JeuvESqF{peudG}KJZu0J>^4*9*<+~As%6F4@FO}~m?_Mh3jTltEn=ovZ?<ViwD&I}sy;Z&&F{peuVo>{Fkausj4+eSnR{LNe2DJ|cVo*9s-o2Ghl6N1alZ4?YokR>yC&9aq(@F5|<8+cRIGrR6N+-#?kJ3r<?xS=PF({n`4D(X?Zt(7TseCth_q<fT8!@PSH)2rvZt(7TseCth_q<fT8!@PSH)2rvZt(7TseCth_be*kO&E&GcOwRs?*{LlMdiD}yJu1PZp5JS-H1W$g8|+>i`oYRyn7b44+dgT`(O};s&o>(dsd~B;N7z-okR>uClQ0w$qFktooujz(@DbMbdoSAodoZmRp}&ocT+k^7@E>a#Gvxs<lRl>yUDwo%6B6MmG4FjD&I}s-BiAtyt}D<H)2rvZo<%2zMH(et9&<kcUSpt#Gvxsh(YDM$-BGCcawK_mG4FjD&LJ5)IJ#G-Cga2LEb&oJ{W{ysC_UHgVIUz?xA#&yn85}L<~wN5rfl7@b2Ms61;mjog@rSCjrA$I!WF=l}?g(Po<NHLFpu7Q2B21?y2(K<lR%{yAgxRcOwRs?<Vh_D&I}sy;Q!NFf5htMhq(7P2Rm!zMH&zseCtLQ2B1epz_`1-Am=W$-9@zcOwRs?<Nde?Snzyz12P#<lS5CgMk>-J{X8W=_GmgRys-Ey_HTP2BnjT!RaJ;_jWo7-hG@-0*2#sk}xQpB=0^-C&{~y(n-XibP_SBd^dUbQTcB2?xXVEh(YDM0mHgfz8k!IT`J!V-n}lB??wzN-;Eelz8k!IT`J!V-n}lB??wzN-;Eelz8k!IT`J!V-o1*-cN2!9^4*9*?Slc{y^7ig1H5|`wGRejQ2SsY2BnkW-K!{_1n*u&=_F!MI!PF+)5!`eIGt>;g40RD;B=BOD4hiFURCKNc=xJGClQ0vNyMP?-QeA;s(d$jcT@Rp!q8N{8!@PSH+gqc`EK&=rt;m0LFKy<gUWZ4cQ=*qChu-4-;EelzMC*~mG36+?ke9+-rZHc8!@PSH)2rxV32orwGRe)cUSvhAO^J$24YY;N#5O+PLg*IrIUnVD4j$MPA9>;hto;$?%{NjFgTqg3`!@-yNA+A^6sH@5-}*9Bn(sKyUDw!%6F4@PnGXR3@YD^7*xKSynCvAH+lC|`EJCZ^4*9*<-5tdr^<JecQ2LiCJamEyAgxRcawK7mG36+UMk;>7*xI+F{pho$h()?2ZOwOseLdIgW3m!Fl?oh<lS58BzgB%I*AyRP9g@Uli=Ok=_Gjfb~;HIoK6x3rIX~{Tj?Zu_fa}Y7>?3O#Gvxs4Sp9#<+~gFE{@7~BL<c4Mhq(7-RXCs^4*<&7b@S47*xI+Fzid^yIY49RKB|nSV85x5rfKiBL<c4?iE&0`R?9e1(okc3@YD^7}P!(ZNds_AB>j&hS^2!gFzUI+6MzMD4p!U$)a?!11F2pNyMOZ5-}*9Y#UZkI@u1apmY*3D4iq>Rq130PFAIp9XMH)P9g@SlZZj(yTQA6Rrzl4?p;;B8!@PSH)2rvZt(71Rlb|NyQzFPVQ4DfjTltEo4mWJd^dS_Q~7Shpz_^_LFK#2yPL{)lXo|j??wzN-%S|0+6RNYyQ_UL$h*7R2LmyveJ~J%(n<2}u5^;TyDObU3`!>vgVIUz?yhu_yn85}Bn(68Bw|oHN!~q_PLg*IrIUz3=_F!M`EK&=q4M42-9zQO5rfKi6Nahs-Q?X<<-5tdr^<ID29@td3@YDE-aS>mo4k9fd^ciH`EJCZ^4;X!Q{}tKyO+v$6NaVo-H1W$gF)WC)IJ#G-AnC*ff&?27>GgXBzgBzI!WHWlujZBrIUnVE1e|o-byFQySLIw#GrH%F({oR@7_u$$-B4GNyMOZ5;3TJH+lC~`EK&=qw?K^;i!B!Vo>>R^6sPZ-Q?Xz<+~As%6B6MmG36+J}Tc$-hEWQ8!@PSH()rI%6EfzpG)Pt!Mo3;^4*9*<+~As+6M!?`&?=t4DjxAseLdIgW3lJF({n`?>?8(N$~Ddlui<cqI425D4hiFK1Jyyc=stvClQ0vNyMOZ61@8qrIX;@rzo973`!>nLsj{1@a|Jpz8k#zRF&^W3@YD^7*xI+y!%v@?*{KaRpq-8gUWX!29@sy?><%KyUDwo%6Ainrt;m0LFK#2yPL{)lXo|j??wzN-;Eg5J{aWPP3?n0-rdwb7>GgbgFzU&(n<2}u5^;TyDObU3`!>vgVIUz?yhu_yt^x%L<~wN5rfi6^6sv5lDvB;og@rH=_F!M`EK&=q4M42-9zQO5rfKiBL<c4Chs09-%Z{<RK6QAsC+kJm@3~*-aS>mo4k9fd^ciH`EJCZ^4;X!Q{}tKyQj)`BL<c4Mht2n4D#-&_Q4?UUTPl<!m!jn7>GgXBzgBzI!WHWlujZBrIUz3=_GmgQaVZAy_8NO2BnjPVJn>^@7_u$$-B4GNyMOZ5;3TJH+lC~`EK&=t@7Q7LFKy<gUWZ4cW;&NChtBf-%S{f%6B6MmG36+J}Tc$-hEWQ8!@PSH)2rvZu0J<^4;X!N9DT_gUWYj4E4IyJ{Z}%A68KNU}W!pSV8TBff&?27>GgXWcKcd6_ie9?|xW8=_F!MI*AyRPG;|ZSV8Gz_U?KWrIUoAD4j$MN+&yTvM8PGz{#R?5-}*9L<}n5oxS^E1(ok^pixETyAgxRcN2!H^4;0HA68KL?(E$UE2w-oVo>>R#Gvxs4K%8%e0Kwlsw&@&7*xI+F{pfZ2ToR%?<VhVD&I{Qn#y-02DJ|cd3RI$V32n=wGRejQ2SsY2Bnka-A(Bvd3RGfi5Qel5{9mHlDxYsoh0w>N+%J6(n-XibdtQgE1e|o?n);SgVIUFkT_Z1O@M-B#WBv!0^nG2ic`Bo5QjK73lM^8ZUN{RM>sbN02|{3=jIj&!7vLDf@y96tSXLgZWaKiiqo5$TOb6-EC2}N?B<rrK*cz_xm^Lc7-u)PWF`zvtw0#MxeKtVIJ&u609-0gZf@>@5G=C*AyB$lfR57%Bk*x4Wh4yCNh4uU+6Wf4ls1AxEv1cwKxrc(P}*34j!TJS0X{CJj)+0YBVmX$o3fl!$C1s=0-#KBVpE4i(98mapqpC&r;g*An+3qB<Fw}H76`#I3lM^BZUJN|j%sce09A^Unwwhy1aVMvvj8EeipoI6IH$Q?0k|0FG`D0X3`4Cz7^b-kkfk`Lxmf^IDNbo_?tu^-vp_~rE~Sg$)8%r)*kA>vjEKQWBVkb5IAH;$jSCh~+DHhLHWC7*jo{PeQsM|cU5Zji!cdevB8I4sTRC-!<C&WU3M>%SaY_WkEI<gRxrGf2EVICY1-7{bLU7CifFRCgZf*f&DUM}s764U>Q<*v?f@T&V1YJ=Xs2FE5w<`b_<4mSbi(#o12*Wma0kRZFGB*o=D#eLRb>D21IFPARB2c<WKHZ!!0w0@FM#SKxkuWH21gCCF8^NiY(ndm{v=I<=rH$m%U5O+4bXV$#7?eCBhNzFnIdvSx+$;df6elrtNd(I*KnS+E1#s#-hdIuv^Bl%~vr+OKW=RCq+ycl_p2HmH)OilGB!XcUAOurU8K{^CF-Q7zp2RGR;iwe=L!QMP=hS%=bDUG>Nz9T6npuDlC|x9<o=zBnk5eflVsO$(7?d`GQ%|Lh;M7ZLBOq8x8wr8ZM)K*U#F2b@DRo2)N*)nI<m2U>I*wv)764_+lNkBUMjph-Z#Lo_=H?c_spA~xW&v>OJcp6rY~(qN{AMH0VQy{#WGT*JZWaJl%5xa`%|@QX$Zs~{9OjnFK*czSxm^Lcm?tszyNx`GvEOaPS<KB{fGovP%*_IzN_i3^zuCxx82Qb{QMyPzeVi}?ACFQ-#NebcW2l$X#t923ZCtQ`(ndm{w2=@fZ3LgLml8+t>3S)3L<~wE5kur-QBGasDCTB?tV}geV%#?yH4kErL{QBwY*?U~1r99G%`FguVHO|+)7(OZ1(sO=R4LD4jzn<G0)QaSVQ#4mRE&d|+ZBL|c@lFhhNe~^4BgxX$Wk1|+$;d9lqWGqB3NbtLZEaJe7aUAjKIgHl#wtrCyj(bX(KpwQ`!hl-IO*G0;P?FKxrfSbW`F;KHZc$A_gUogdxsiYIRN>M=>`GfHK8NOuHn4W)>g>-P{5=b)3W8EC5a&=P>P(2$orZ5NvY`AWLx$bF%=bQk=sy_svF)bC`BX1XWQPs2B$^w<`b_<0PhC7Q;|05Qb^)0%R$UVr~`yRf?0Cc1Z-sEC2|m(na#=>4XvZIF&LY1}BY#L1`m6^;FsjPCb=25(1@-gg|K{`SeucNIt!kIueGZ<PkAMeQeID<0$540Z^tmiD~kijW~#D@|%r3hiT5K^Bkr*r;c-&CcoK;bC@Q-*~oL4=A1guVVZO5IEQKSn~gY!Y4V$mJcnu0r}H4DNuQ3Bm}bA*h@+ThzuU;OnC6^1k7Am0>NtsM@|%q~h-vbhjiYpteEK+H1U??6jEKQWBVkb52u^*JHiA<hrHzC@X=6svE~Sm&)9q5?2tM5|rH+U}$s=NLS?Ucds4#Uuu!2id69yHhMhq%XJtnN6^3(%3SyY~yFcg)iMhq%XJuX;51*%7Z6;z@cF{ns2Vo;gt0h}x<Q$2u_MP;fHgUVDBhN||Q7!_7fdrpi7E2upu5QExt0x>9^oWRMdbaDbGtI|otpmY*3IGqH~s!k`tvzpUMz|fpd5(c;L1aPt`ot(hQ=60SS3`!>vgUeTgMl}_%Chu-8VNDoR#F{X4m9Zx8?kZzV-rZHk8ZoGhHDXX1Yx3@{Le}KnU8SrMgNj)r29>iW@9rvRP2N3J&YCa`m9s_+Y99*n?xFUfAnzV(9}2{v_Mt!wN+-#?htf&%?xA!NF({oR4Abc(c=vQV3En-OP7(&ElZ3(TMFE_gN+-#?r`wH!Fesfw3@&pG8Z}kun!J0t)HPsODt3(+RPLI*d#T(tdG}JeYs8>(*N8#ouF1QX3SN_UFO|GT3@Um}7`Dn@lXq{Gy(aJ8DtnC>RQ4J%sJ$u3ySLh#g1mdHy(tib+M5C~D4itl-byFQyN}XI!f=#MA_k|E;N8dRBzX66I!PFuP7(&UKLv2|D4itlK5mB!!k}~#F!amius5urg4p2Q{c=fc!l0tqh(Tqs!MppVve@9={Zd(M#Gta+h(Tqs!MppV!r0*5y{I%cVJIq&jTlrO8@#(0mB$9}?nULX5rfKOBL=l!1$cKaYQGBb?q1Y>6^KFYS3wx6(n;{{UX@OQclWAv5-}*9L<~+RJFMVza=;2sCkcboNy6avtN>0{rIX~{&FxwN7@E>a#Ne{opixbQv&p-gOJ@@X70*TtDxXc>-BdoCyt}D<HeyiuY{Jl0KAXI|tAI9ncUK8*#GoSDh(Tqv$-BGCXp?t$mC;5FDx-}U)E*Y(-CgZrLEb&o9u|aQs68wYgVIUz?xA#&yn85}L<~wN5rfl7@b2Ms61;mjog@rSCjrBB`&a-cr_xFC?&)^2APh<;5rfNXgGNmi*e36uF0oA*RAd`5sLVEb_f(l}^6sTF+k|1M%r;_BnQij!r9#`}-Akpm5rc|tBL<b*ChuM<w@u!?RBjtFsN6PT*lJ%3^6stnwIJ`_YF`V)p!T&u3`!@-ySLIw^6ss45-}*9L<~+R!MnH9N$~FDbP_Ner;~)i?QH>^JW40YyN}!5f-oqZL<}y&4H|V+h?~6oxD+>GP%&=6FfNtj2Jar1%5j5tk4xpa5rfKcBL<b@2Jar13UY&Yk4q)F5rc|yBL<b_2Jar1%5sBukD{{NgrTS`H)2qGU4VCwqV~D~?;b_%b%7YvUKfZ#=_GjfC`u>6yGKzvi5Qel5{Bw@vcn2aCkL$HbdoSQog@ryzYE}GRXPdYJ*wOBf-oqZL<}z14H{Kdu$#QQxnwtBXe!!`7*w{Kyt}DvH+gqc*>1$3vfYS5WxL6{n+kW6cQ=*pMhq(6O&Ge$cawK_mG36+?keAn7*xI+F{u47$h*7R|AM@`tNkw!gWCTBF({oR@9s(`$-9TrNy0FcP9g@Uli=OM=_Gjfa5_mCoK6x3w+9Arawwf7?;dU!48ovvk}yn_?=Ik3Q{}r0c-B<;Zp5JS-H1WuyDM;Vs(g0^PEM8YMhq(7jTltEyMSj+mG3U#Sxe=+3BywPZp5JS-4!^wRKB|cCzr~1BL<c4Mht2nj0&E$)IJy$JZq_aFc5><2ZJzdrIQsnxs^^<;N(_1i5QelA_k|E;91-0BzV?#I!PFuP7(&ClMOhzl}<L`<WV|F7>?3O#Gvxs<lRT*yUDwc%6B6MmG4FjD&I}seN?`iy!)tpH)2rvZon`vmG1`co|npZgLltM<+~As%6B6MmG1`co|npZgLltM<+~As%6B6MwGRe(_q^0T7~tKrsC_U9Ls9!+AO@wA;N7z*odoZmMd>7BP&$bioK99)!Rcg!6`W2I2B(vNp(>pO@19lZBzX6%N+%J6(n-Xi^4;Lwv#NYIc=xO--;Eelz8f*9d^dRatSaA4-rZEbn=mw$??wzN-%Z}#RKA<MyQzFPVo>>R#Gvxs<lRl>yUDwo%6B6MmG34DUG0NG-rdzc806hu?Sp|B)IJ!9LFpuUcUL+|-rbc>A_k?Ch{5S3cz1U?3En-NP6CGEbdoSAoh0ubN+-#?htf&JpmY*3sC+kh_fYw6^6sJX-H1Wuy9vWo`EK&=sq)?A-Babe5rfKiBL<c4Chwjq-%Z{<RlXZBsC+kKQ2B21?y2(K<lRf<y9vWm`EJCZ_Q4?UUTPl<^6sVf!9WaZ9}L8xbdtP#DV-$mUP>nsgVIUDu$@kVcW<YY;N9ElBw=tmNf?w)l6P;VljPl7=_F!MI*AxmzMH&zt9&<k_fh$7!f;f+8!@PSH+lC_`EK&=qw?K|LFKy<gUWZ4cORASChtBf-;Eelz8f&COXa)4yVs@i-QeBpQu%Jgpz_^_LG6P9-n}lh4+ePmy3{@xh(Yaxff$rdf_JY==_GjfDoQ5_Ls2@37@SU4Si$LJgB6@k5(cM}ghA;fc=swwC&9Z{Q96kjlui<cs`B07-K(m6H+c7|D&LJ5RK6QAsC+kg_o^!24c@(~%6B6MmG4FjD&Gy>y{gK0lXo|j?<NdQ<+~As%6F4@H<j-u?`|sJjTltE8!@PTFvz=`+6RNYyQzIJ5QEwWgD`ZZljPl9=_Gk~S2~FplujZBr<35_-RUHFcXv8T7@STL2Bnka-CgM<dG}B{Nf?IGNyMP?-Q?Xv<-5tdhst*&29@td3@YDE-aS;lo4k9dd^ciH`EJ56Rlb|Nd#ZdldG}QLZp5JS-H1WuyUDw!%6F4@PnGXR3@YD^7}P!(<lR&4gF)WC)IJ!5VX1vE5QEZ5^6sT{lDvB<okR>uClQ0wN$~FFbP~LKIh`a7PA37wRys-Ey_HUqcW<SWh(YNjVo>>R^6stj-Q?X{<+~As%6B6MmG36+-YVZs-hEWQn=l-e??wzN-%Z|qRKA<M`>1?3Vo>>R#Gvxs<lRT*yUDwc%6B6MmG1@&`%?R0fOqdp?Slc{y)U&724YbAU?2vili=O^QaTCVy)UJch(YNjVsJWHVFjm?*}HdfItdtx(@DahbP~LK7p0Tn-Mc8AL<~wN5rfKiH~L+ue0QVYh01p$29@t73{~a3JNz!H%6E78T~w9tMhq(7jTltEyLDJW<-6N}6;!?(F{peuVo>?+4xFqi-`#<eP35}@LsR)~#Gv-UXcJaY`(QNq8>Xp!Fc5><2LmxEo$SELrgX9cC!5kq#GrJNFm$Dp4gQAdN+%oq4bzoQA_k?Ch(YOO2TpdSlN~tOl};iCrIUz3<-5tdyUKTycMp~CCJaO6yAgxRcawJymG36+9xC6B7*xI+F{peudG}EHZu0J-^4*9*<+};PRQYc5?y2(K<lR%{yAgxRcOwS14+eSnRQq6%cTcqs24YbAU?2viljPk~=_GmgQaVW(meNVYpmdVFdnuhH?_NqL5rfi6#GrJNyn88~B=25IClQ0vNy4yIzMH&zt9&<k_g49C#Gvxsh(YDM$-B48cawK-mG4FjD&LJ5RKA<Md#ijmdG}HIZo+U>z8f*9d^dUbQTcB2?xXVEh(YDM5rf(XgS`8weK5$okJ<+VF{pho0K>VIPJ(xzOX(zd_qmi#A_k?Ch(YNjc=x%KPJ(xzOX(zHP&$bilum+opG)Z^c=stvCkaDQI*Axmz8k#z6qWA=?><H4yAgxRcOwRs?*{KaMdiD}yH8R1Zp5JS-Grg4d^dRasVd(M-hHadcOwRs??wzN-wob<s>*kRcb}^A-H1WuyAgxh2LrtORJ9KVd3RI$U=W6;_Q605N+-#?o6<@0?xu7SF({ox3`!@-yPMKU^6sW|5-}*9Bn(~YBzbpNI!WH$l};iCrIUz3<-5tdyUKTycXyTVMhq(7jTltEo4mWLd^dUbQ2B1cFjT%9F{peudG}EHZu0J-^4*9*<+~As%6F4@50&pG?;a}OjTltEn=nkZ4+eSnRQq6%cTcqs24YbAU?2viljPk~=_GmgR62<mlujZBrIX~{Q|Tmm_fk4Z7?#pW#GrJNyn88~B=25IClQ0vNyMP?-Q?X%<-5tdm&$h|29@t73|r;9$-B48cawK-mG4FjD&LJ5RKA<Md#ijmdG}WNZp5JS-H1WuyUDw^%6F4@AC>PW3`ga=5rf(XgS`8weK5$okJ<+VF{pho5QEZ5^6sN_lDzvUokR>uCo_h2T}mgjcR#G4bTWJQ!wO0#5rfi6#GrICd-uZ%N++{-Kdhj15-}*9L<}n5oxS^E1(om4-rcUE^4)}?sC+kKQ2Flc-483Me0TQlhZR)58!@PSH)2rv?(E$UE2w;T1C1&w-;EelzMC*qmGADr$*S_*9XMH4z8f*9d^ciH`(QNCsH*nCXrNJ5?Sp|B)IJ!9LFr@%PFAIp<lRl_Bw=VuClQ0vN%HQdbdtQgDV;<NN+%J6(n<2}rgW0LyD6PS3`!>nLs$83^6swk-Q?X}<+~As%6B6MmG36+?ke9+-rZHc8!@PSH)2rvZu0J~^4;X!L*=^(!%+Ed#Gvxs<lRH%yUDwU%6B6MmG4FjY99>p?xFU<AnzV(9}L8x_Q4<wQ|Tmm_f$Gb-aVC0A_k?Ch(YNjdG}O0N!~q`P9g@SlZZj-BzgB#I!WHWlui<crF0T8L{7f@R}ffM9OK+90FD)>IQ^0chFO3ROmhoB$2h{dSpe7=Cpb5^KnRXm01(9a&CM-<RmJhm%>v+5ae8xe3xuGV1qeY`R0b->+0E?=z{NPbxg|4USZW2ru+3e7O~uj8%>v+3adLBW4}c&JZf+JJ1WFeR(D6871U??6jEKQWBVkb52o`meHiAPPrHzC@X=6rEE~Sm7!wO0q%YYS>IwA%okBA}aV^dCD;>hM^fdUIebu1FWG7AubZEj)10>><now~$n&CM+Uf;g<XS%45!a|;y~Xl8*13v_b}gkYEj2*FfT1}erm&Fu=n#W<(AB{N|-Y6ZX$=QKBW0kRawG&c)?D#a<y%{>r;W)>g>N*BSWOLf8se5^_t5rdOP!l1Nq!U9Sg|NrLBZ8x&pih}O}@(+TZW8je7ydfVWv_~CB$w->$Vqg&X-_w#k9v#Olmx0DWPe#T^bR9LlN+z{-z|=)$BM=mojYOcbkvv^gIFhG}N=IZ+@rVq|d<<vm<@L;N0Z69h?M!4zFwFu$u)Leuy#knec{Q_J08G8SnTae3idldN!n^{+((+nnw*aKl@>V9YB<N-VA{c5b!&Y40$?R8vy|}!S33;}WmUl9dEkibM0kO2alG!Z)skFS2i7W}KS%3&sE|RCK3r5(FRV5=bxM&21rm_)C-BdP$shi41B2d{#1S%WJ(@ljVdAg}|L<SX)$gs@E>P)@7irFmy$+WzQ33;-yyod>Tva!C0sm|2vdzk7>y}pN$CmZW~7<sa>zK5yK)a!eg>P)@9hmj{6>w6e^va!C0snXNyi<l}sy}pUDXB+FQ7<;y{zKf~O)a$F5>P)@9iIFE8>x&q9vN2RHlBb6YM%a&2B_lCR7mdWAvJp%@RW^dDr^-emP}xWXDjUhuQ-vdWda86p1{IIMkk@xHO_@5cuVR`qbza}Zgd`|t0U`+V3Ii6XW`PL{H1i5X(9HrwFw84tSYVn3)~S@&_b~2cBd_maLK0-Pm0>HcFJhYWbY9=Ygk`8|1!8FCEdmzkW&ud0^-WAjf@u~2f~;~8Je{)(M%a&8B_lGpXe0)ejSUu1+1OzLm5oH8vXKZ>HiD;fR^dpVE-D>~p{RI7hGjl>XX^D;On0VU-^5f&P|X5F(9A1<sn_=~-I;oQ4^t(<Gz$R1`W~h`Q?Kt~x-<3q9;QlyViq8RP+J+c;`$<{OHZ$FVyb26Y6W5#<}DzW)>kpznR<N_<4!j6`XZ)E0+oy8>FR<J_G4Aahzu?oi9ux}n7XQL1XEX)jYOcb5eS;fM)Gu1;YgltDjktQ#UnB-^Km#+udiZ;Gxho=MxJb}FJk1$#`+#+I8(3hVTLpH`W{A}Y^?8L<jKbR9%eXGukT@oGxho&MxJb}?_uQ0#`+#+NKdaXVutke`X<JnZLF_i?AgZpE@n7WudiZ;Gxho=MxJb}FJk1$#!$IPo*ph3VLuL)jL6`kkr-4qf~kkfMlkhM*$4zvWg`)&Y$Q)l6^`WTsnQV{R6HWXG9RZi_4+DiI#aK2V&uuj`Xa`iY?Spq%v7c>>wB1~OkLLZFkKQ9vj7o<d4&uMRI@;V1)6yUBIsrTA{c5bZ?M8tE5Ke{-^9qXjk3Oq>6Rgzw+L9Em<1{<5au0-pqd4UK;<HMx+E8jupg63Mr3f&2n<<eV}k`$Hg;G*Wg`)&Y$O7ejo|5$RXBpDOIGQK3@RRx!F8!YQ(2X%13Xn+n;IC3s#7C_>Qh(Pl125YE9}Xl`qapv`qapv`qUL%RaBw6f~$&ZR3n3`R1-s}PIZMn8LCrVVNZtYR3n4xR3n3$IZ?q^p=M50@Kvap6Ud-uP9THI$p(8eR8BV7lU3y;F;tb4$l!7koK;;;g0rg2Nn&t0Nepi81nkMGa<ajmtZwoIF{qp*hUWU!kWoz)tjXQYHLQt2RjiRgb*#zVO?9lv-A#3@kwJB=kwJB=$=ywrtjXP7wXBJutC}@3sGc>syQ`ixxx1^LH8QB4H8Q9<6y)x%=1`EkyP89R3~CMqF$|TH<nEzzlH5I1P9lTKNn~(23GN;)C&Asr<s>n<oFoP}ivsrKP&rBNo^BciFie$`$lyBHkWo{WuF2igwXTUl)vl32^{&a?Q}wRN-Bb0hkwNvYfgv{4y9Rg1ri$0#?$}iG8W~je8W~jg8r&V5>RyApV^iI0WKi8}WKgpyz}>N_*%aXJNNP3(F(fsc0vS|Jg1aNBoCJ4AQaOnXDkqV_<z&DLE+;Fj;Bt}}TuuT*cJnD<PiB>q;O@w7LIp9XoJ0oK!-kB?s)!Blj_jJ)#GtC!$e_B|;O@w(i%sq>s*6nwMRl=}L3Odo-9?qL$=yY@v5`U5v5`UbvB}*<^|8s_MfI_fLG`hTA=JDIa(AeC73A(v^D2-*&8t8Lm6PP|P&rBN4waM0pmGu!Tuy?!!{sEnySkhNhU#*X7~ISX*ppS|B)Pk~sTIVaauOL_Hybjls&Y2DySjEZF{pYrF*McBCU-a0&n9;_)z3x-)z3x-)z2n(H&xIkcQ@70Mg~>UMh4Z<CU-a0(I$6y)zKz~t~%Ptpk`Q*yStiULGJEqh6OUH85YQ(a+2KLRZfz-yUIyqP&r8q!{sEnd$^ngcMq47#Ncw07~C8S*pox$B)NOINfyMQauOL_UmG%NsKPe6d%DIpFich1Mh4Z{CU;NO*(P^S)!9Y{)!9Y{)!8O@PgU9`cTd&YMg~>e28Oz+-Zr?qZmPEp?yj5aZ6kx~Z6kx4YXR=Ao0@9@?yj4fYk>@Et_3ovoCJ5*P30uGyC#*B#E?`@B7@7xfE8R$R#?I1Br&+0BnCIz0`_E5ISKBr$xXK)29=Y<kX?rxGAgSQH@LfI*WxAyRpUkm)#C<t*Q|Qn;O?4Lj~f|Oj~f|Oj~m=wvnq0vyNhaa6GKr|Ze&nhZgO`~U2bxBQC)6iP+e|hP_r(`-9^p1Aa@rv>jD|ntP5fYm6PP|P&rBN4waM0pmGu!Tuy?!!{sEnJ6ui@gUd-`aPux;Pln1#a(8tTFMy${oJ0oK>xPV~s@P5LuCCcl4653V4655r?yjoaP42F$+l>sW+f58jb-T&kO_jUJ-A%Q-kwMkFkwNvl$=yx$yUE>6^}CTl^}CTl&A%XbH#Pr)+}+ju3u5SM{sl6qoFsR5m6PP|u5uC?R8As;%Smu|cR2~}?k*>Z!Q~_{3^xM<_T*4GN$wtQ3I;K#oJ0oI?@r*Xq59nkoHbOx8yQr;8yQr;JHwtFs^6VqPfpeECWfi{-N>N&-3gpERlhrdv!?2IBZKO9BZKO9XV{Zd^}93d$*KC?$e{Y&z|b}|2P1E=f|`So6Re=-U?78<gMkbxC-Z<6R8HmzE2x}A29=Y@;Bqp<3N9zt&T7f!BrqhGlf<BMvcR59Dklr<$)s`;8B|UpgX(vKyIWHIZg6)?s^5(as^3ivS@pZY-7TwrH@LfH)$c|I)$c|I)$ay(x2*cz;O>@HzZ)4;zZ)4;zZ=}$vg&t}yNl{~6GKt`Ze&n%Fv#6S&A}jd7c~b18PpsMWKcOt?k*}P$=yZeBr>R+B!+M~3GNP;li==fIY|sICy7DjB)L0OPLjJr<s>qwoJ0oI?<RML>UWd7tLk?XLsk85WKjKXa(7kzZgO{3{cdDX{cdDX{cdu1RsC*qcUApvWKjKXVrZ)0P3~^0-%ajrs^5(as^5(aY7PduyQw)C<nE^CU?78<gMkbxC&}GS<s`YgtDGc;u5uC?Tuy?!yUR&%cXv5S3@#^$LFFX5yQ`cecXySO$e?nP7>4S1le>rNcayt^>USf9>USf9>UWd7hw68eyNBv`BZKO9BZKO9le>rNcayuP>UR^vRQ+ybQ2lOl_f-9Ea`#mIZe&pXZe&n%Fv#6g&A}jdPc;Vv8PpsMVCb95NpN@HR8E4s`=)Xd8B|UpgUiVbE4Z92u!74;VsJT03@RtV-F;Iz3GVJm<s>mAm6OPz`rY8}o>adZ+})GvcO!%9cO!%9cZ0ioQvGglcTcL{jSQ;aO$=G}yTRQ(tA01QyJyw!Mh4aIMh4aI26y+Y`rY8}o>jja8C1U;8PpsMaCgsY4hFfqs5uzKP}Cd@WKcOt?k*}P$=yZeBr>R+L<X0W;O^pb65L%}P7;I5Nni+-ljQDDIZ5sgm6OPzauOL-znk10s^3lS4%P2Q2G#FI2G#E-cZceCle??xcN0TZ{cdDX{cdu1RsC*qcUApvWKjKXWKjKXa(7kzZgO{3{cdDX{cd7tY7PduyQw)C<nE^CU?78<gMkbxC&}GS<s`YgshmUxm6OQeauVF#Tuy?!yUR&n=q@LTLFFX5yQ`cecXySO$e?l(8C1WU+}%~bo7~-1zZ)4;znd6_>UWd7hw68eyNBv`BZKO9BZKO9le>rNcayt^>USf9>USf9>UWd7hw68eyQk`R6T?*fZe&n%Fv#6g&A}jdPc;Vv8PpsMWKcOt?w%?q$=y@sBr>R+1ctG>oXoI-%gF*OxSS*gmy^VxauVD<HkFg$?y;$yL<W_U$e{Y&;O?=hemA&#B-QUGhNSx4$e{Y&;O>!BzZ={=lInLOgX(u9gX(vKyGK&}ZgBTVs^5(as^3ivS@pZY-6N}hH@JIb)$c|I)$c|IH3tLSJ+hjE0q!1I&A~thH3tJ3R8E4sM^-sW?k*}PiJ_>RL<X0W;O^pb65L%}P7;I5Nn%hrN$xHxC&}GK<s>qwoFs-&{q6#vMW}vvfzKjTzZ)4;zZ)4;zdPWw2-WWn`Ycqx8yQr;8yQr;yTH#dq59ngeuk;4-%Siv^}CTl^}7T1WL5p{fIV4NzZ)4;zZ)6U9E<`#!&Ef~qrlHFRn5Ub1~mtR7@Eq-fIZn%P6q7Brg9P)R8As;%E<yh!!(tX1%8HUDkqUa<s>qwoDA5LP32_3p6n_oiJ_~UL<ZIGCU<w$?<RM5)$c|I)$c|I)$b;Ech&DEcX!qAMh4aICWfK<-Q@0}`rYL2q59p(p!(g&p!(h9?xFhK<nE#R-N>N&-N>NkV350qnu9^^o@x#TF-$cF0~u6KlDntMNpkm8If)D^Cy_zrB)NO4oFsQom6OPzauOKkrg9S8JvWt;;O@DpoJ0ndlgOa@-Qe!IseU)Odv2=VjSQ;ajSQ;a4ep+s>UV>?XHxxcVo0jrjSQ;a4ep*v^}E5{GpT+zGN^txGN^txxO*nm?*@0zr25^+p!(g!kkuRvaQDn=4hFb;W;F)`8PpsMWKcN??w(oYB)EHKm6OPzauOL-PJ+8<Ryj%TE-EL9p{Sfh29=ZK?xJ#%++9>oB7@3FWKjKXa(7YvZgO`~{cdDX{cd6i)$b;Ehw68eyF>N6kwNvlkwNvl$=#v)-Q@02{cdDX{cdDX{cdu1sD3xOyQ+RSF;vy>Mg}zpgWO%!91L=IRdX<qLCwKH29=ZK?y7Q<++9^pB7@3FVrVKS$=yxmB)Pk(oJ0ndlgOZQlHA=?PLjKu%1LBUIf)Fa-%ajrs^3lS?yBES3|;lRkwNvl$=zM`yUE>M^}CTl^}CTl^}EU4UG=-k-Cgy&kwNvliD9UIH@SPLemA*$sD3vxsD3vxs5uzq?xE&jkh_PPgMkcc4hAx)oFsP-m6PP|sdADSrpifVP&rBNo+>BF-BaZxGN_zH29=ZK?x}K;+&xuJB7@4wHA8Hh>UXc*eOf{FyVveMt)Tkd$e{Y&$e{Y&Yj>YkQ2p+;yH6{qem63xem63xe)rnlrxjGcd+qMnlInL8LsI>2WKjL?fIXR1zdK-0Ce`mo2G#FI1~ms`?e5bGY7Rz$j7n+_1~RBQ7{riOP6q7Bta377PiB>q$e?l(8B|WL-F;d?<>cDkrxjFAB7@3FWKcO7uqU(1Npg2lIY|sf<s>qwemA+hsD3xOyQqFQGN^txGN^txxx1)-H@Um0em63xem60M>UWd7L-o7K-J$y3$e{Y&$e{Y&<nB=YZgO|1em63xem63xIT+;bP;)TI-Br!OAcm^uU?79aNpg2pIZ5uWDkqUa<s>qwoFsQwm6PP|s&Wz;R8A5@Q#ncQZYn3q-A&~rGN_zH2G#E-cQ@7VCU-a0??wjI??wjI?<RLQ)$b;Ech&DEhOYYE$e{Y&<nFHe-Q@1B`rXK&`rXK&`rYL2uKL~N?ymaX$e{Y&#4ywx4088Sb1=x=L(RcJ1~mr*8B|V^yNAk2a`#X<i3}<ykwN7oxqGObBzI4hlf*DpP9lTKNpkm8IZ5uGDkqUa<>db}y!_>-*C#FyPfy3Yr<?n`s~-;^pO06E4<DXypFbX6|LFMR>F+&${b&65%R-lj>+6T(<ICSFC+B=^i(Y=XdOG~@_!jGTBJQWZ(R!oBAFP-D)Os`J^RAc2Io~&Jv7M7Io&F=?ynH>Mmv5Z)`!`FRb-&Eg+j-ZU`K<4I%HXrESGV(i|LS@3VAgWp&!g5q{dv6n<>%bJ{Pb}Duj7Z;c)b5~d^p@)yO>P#?YH^xyRY&4KNd1<dRx^`i~mRSuS?uNKYe<By88a`cs#8$#P=(2F}^W)IS+Vh+q?UYZw5SX<K2OOncw&a<oW$=@-Gt?|A73h<=)9ISDU4L^IR!s+AJk+?~Yt;mO0-l>6?(Jvuu_*N4z_5k>xMHqV%o*^~`74<uAXM6z`6FQ{ZEJci?WX%=h<ir`szjy}w;<67l61zQ()zp8rbz=EwK1=N(V?S8slne>i>V@#_Bj$K%6~uZJHmAS3<qpKi-9d%nB*&-3x;uYUQ8n%;iF_3`@Z!~N~2`@7Q@U;XxL|9yJ?mv83fuTO1Pce~@U?Kg)$Zy(&9f+cuxci0v1;O<PT!h^e`s{s%0PN*h4xH}w88J^r7f~Epb?hZR?$=#VIt+@Z)%0^G_KAlca?mnJ`pZM+rO7xlUfBP}fr@sF+$F%(1({J`>YVhQKb9Q)g_xgE*CwFh36Fj+l`5f@%?%gxJ`)-Hk0Z;CBXrA!oZigmXbGJnkt-0T$In$%NPbn9AbpJ8oN5B7^=(FGN(n25pewTh3YW(&r`}_OnyX(Wl=hJhS$J<X&pD*qoE^ePcKHYrxdboXk@pDdJ++4qe`9D5i+}=Ds-rW6g@o@a%=JDzH@ZHM;FMsfOygi&A`SAGni^Ids(?4&Ir~h&By2|6}PmUKicfSbX@bTm4i@W=$i^rSSbMxc%<zN5u)BgZhs~^$', 'tinlayout-1h32-permutation.json': 'c-ozo%Wm5+5JmR|`VRxG5+HWOmmUewO*i?6I4~%gf`v$=LeX`MApc&<QZ-7!pmkOrgS_KI&hUEPzJCLVdeutp>avgSAS$Icx-Mn&=*+XyH|1Vxt(qrT{eW(N>W3vLPELWTs$Mxb)^1;GFj@hPdA2&Qk%wDG!PSk5ZpPo(%TD?HkHa^Bgx`z-qr>OZw8<Q7IqZt8)X%HZw4GIbuXZp|^znMli2r3!ud;MNA8r#4h3xAdI%V5~ldd*8TBh4p1BK7|^KP>_8vDH2?3%t8D`}OMn~ke=BTwewR&<~6RjoT8E|RTlMyG^+lV6+@qLS<x{Y2vrPDL_ouSmYD%8DdaB?_tXf-o)%&Qr$oB9n2Ni#(I05Gl`!N~CeNqiLEmQnEsb9iI}uw5{BhM*A~6rOT5~{WjjEFJvdJEY5V!9@KQlqr{ry)|u^G-|e}+SQ3Ev=1)ZDh|f_nN7)<^ln~Sq6p=Y%b0p>{oueETC?TjJC?W_#kc2Mak%S@%MHxjITO0ss2;xx0p@>5fhoX$4j4ck1I1F(Z;xNSFh{G2(Kp9`i5QiZSLmY-U9C0|xILbK6IN}Jz5r`uYM<9+sS%R`;aSFr{h$9w#fj9zj1j-VWC2WkIKD2mgnvX3$*+#g@Nub?Xpvm)KyOUr;rlH<kwi6HbP!eh<9(XZH;O#759t%D~5qu^j_+nV-nJhayUNre%Md+PXvW-V)>A!O8|9j>Os&m2ZTvW+RZt{YvfA!W$;MSMc!-6g(f`*iaylH7CanM<4&}4YfZ<2pl&(ba8pmETkhwz}CBtd7%f+iCO{f2&SJ+*eq`axxyJ?`dr0Hyh*Y!9;^!B`>~r<f~eI;i}N0{*YLABGyKA$!gIM5TA;czq93W&h-v)Qqmz34^P`IA>bSa2*Xm@8gtEn1kL)J4^@Bd8D}=hZj&U2ier_1b$?y^<LT9`T44{K9k$4d9Ip<s#bQMYIk1plZ^FS=cxxO+1AbJ9`vF0X@>I|KLbf;JgfCHrJoANiNq<7_Wm0x!UDt', 'tinlayout-1h32-permutation.i32': 'c-k%5(-I&E5(L1V9ox2T+qP}nwr$(CZQHhO`!eG8AwQv`tE>O{=bwK9@Gk)gMsPw9l5m740uhNubYc*bc*G|G2}wqBQjn5#q$dLz$wqc^kdr_JCI~?ZMQFkhmPkY<3Q>tgY~m1?L?k8&Nl8U&(vX%+WF`w)$whARke7VqrvL>hMsZ3|l5&)%0u`x7b!t$Pdeo-@4QWPmTF{bqw5J0d=|*>Y(33(GrU*qTMQO@VmP%Bn3RS5^ZR${$Ml_}gO=(4I+R&Cxbfybk=|yk)(3gJnX8;2k#&AY3l5vb@0u!0WbY?J<dCX@43t7f;R<M$FtY-ro*~WHuu#-UyW(Y$W#c0MbmPt%z3R9WIZ00bRMJ#3sOIgKg*07dMY-S5v*~M=5u$O)8=Ku#e#&J$?l5?Er0vEZ)b#8Ezd)(&%4|&FOUhtB4yypWS`NnsC@RLIv<_JeQ#c9rPmP=gb3Rk(sZSHWFM?B^UPkF^_-td-BeC7*Z`NePk@b{np4}SpuB_P2FP6$F0j_^bvBGHIW3}O<G_#_}9$w*ELQj(7JWFRBi$W9J&5{SSAAt<2;O&G!wiO57DDzS)79O9CQ#3UgpsYp#4(vpeHWFafL$W0#dl8^ippdiI4P6<j<j`CEXBGsr)4Qf)4`ZS;+&1g;wTGEd8bf6>M=uQuMQi#G7p(v#&O&Q8kiON)=Dz&Ih9qQ7E#x$WRt!Paf+R};6bfGJ~=uIE`(vSWOU?9U7&Im>_j`2)jBGZ`83}!Nq`7B@|%UI3|R<e%uY+xhX*v<}iGKj$pVJM>*%^1cqiOEc1Dzli)9Okl!#Vlbdt60q%*0PDsY+)<A*v%gHvXA{7;2_61&IwL(j`LjLBG<Ui4Q_Ie`#j(w&v?!YUh<CjeBdMB_|6Z0a)`qm;V7p#%^A*eiOXE!Dz~`J9q#go$2{RFuXxQH-tvjheBmp<_{|^w{<ru8@Gk)gMsPw9l5m740uhNubYc*bc*G|G2}wqBQjn5#q$dLz$wqc^kdr_JCI~?ZMQFkhmPkY<3Q>tgY~m1?L?k8&Nl8U&(vX%+WF`w)$whARke7VqrvL>hMsZ3|l5&)%0u`x7b!t$Pdeo-@4QWPmTF{bqw5J0d=|*>Y(33(GrU*qTMQO@VmP%Bn3RS5^ZR${$Ml_}gO=(4I+R&Cxbfybk=|yk)(3gJnX8;2k#&AY3l5vb@0u!0WbY?J<dCX@43t7f;R<M$FtY-ro*~WHuu#-UyW(Y$W#c0MbmPt%z3R9WIZ00bRMJ#3sOIgKg*07dMY-S5v*~M=5u$O)8=Ku#e#&J$?l5?Er0vEZ)b#8Ezd)(&%4|&FOUhtB4yypWS`NnsC@RLIv<_JeQ#c9rPmP=gb3Rk(sZSHWFM?B^UPkF^_-td-BeC7*Z`NePk@b|yaAAo-eNHBsEf{=tGJQ0XUG@=uOn8YJK2}npXl9Pgzq$52U$VfJ_lY^WDA}~P+N+?1ThOk5;GEs<1EMgOfxFjMmNk~d6Qj><XWFj+J$Vx79lZU+IBR>TwNHL02f|8V@JQb)&HL6pCn$)8{4QNO+n$v=ow4*&8=twuZ(}SKAqA*1$N-0WHhO$(mGF7NbEoxJTx-_COO=wChTGNKMbfPm|=t?ho(}%wFqdx-}$S{U8f{~14JQJA6G^R6ynapE83s}f9ma~GDtYbYJ*vK}vvxA)sVlYD($|y!NhOtayGE<n!EM_x@xh!HaOIXS(R<nk+Y+^H8*vc+;vxmLxV?PHt$T5y{f|H!%JQujgHLi1mo803*4|vEkp7Vm2yyHC|_{cZD^Mjup;xI=z$|+8BhO=DaGFQ0DEpBs%yFB7CPk72JUh{^xeBv`-_{uMS^M}6y|9^l7;9mj~jNpVIB;g281R@fR=)@o<@rX|X5|WJMq#z~fNKXbbl8x--ASZzcOb~(+iqM21ERl#z6rvJ~*u)_&iAYQml9Gzlq#-St$V?Wpl8fBrAusvJPXP*2jN+7_B;_bi1u9aF>eQen^{7t+8q$pBw4f#JXio<^(v9x)peKbWOc9Dwiqe#!ES0EC6{=E;+SH*gjc800n$n8aw4p7X=u8*7(u>~op)dXD&j1E8jNy!6B;y#*1ST?#>C9jz^O(;97P5@xtY9VUSkDGFvW@NRU?+nZ%n*h$iqVW=ER&eb6s9tZ+00=si&)GOma>Y~tYIyi*vuBTvWwm9VK4jG&jAi{jN_c(B<DEK1uk-p>)hZb_qfjk9`cOmyx=A8c+Uqu@{RBO;3tPT%n^=qiqo9oESI><6|QoN+uY$Uk9f=zp7M&<yx}dM_{<l+@{8a6;qQOtTLAtgAi)Su2tpE$@I)XY(TGkAViJ$|Bp@NlNKOh;l8*FbAS2nxP7ZPsh`<CPD4_^V7{U^X$V4G3v4~9^;*yBOBq1rONKG2jl8MY@AuGAaO&;=+kNgy%AjK$72})9q@>HNA)u>JlYEqB-G@v2PXif`S(vJ3Ypd;PrP7iugh{6=1D5WS(8Ol<L%2c5$wWv)U>e7hDG@&W2XiXd1(uvM=p)0-UO&|KwkNyl`Aj25W2u3oF@l0SM)0oZ-W-^cYEMOtaSk4MovX1p^U?bbu&JK1mh`|hDD5Dt77{)S*$xLA?vzW~s=CX*zEMY0DSj`&NvWd-XVJo}X%^vo$kNq6rAjde)2~Kj3^IYH}*SO9NZgP+NJm4YEc+Lx6@{ad>;3MDo&JTWah{GJ=D5p5h8P0Nv%Ut0qx46w6?(&GoJmD#?c+DH$@`=xU;VZxR%^&{$SG)({Ujh=0;DjI~;RsIzA`*@0#2_Z|h))6%l8oe}ASLNYPX;oQjqKzgCxHk|5P}kl(1al@k%&wbq7sYP#33$;NK6uvl8V%%AuXB6Oct_|i`?WPFZsw%0SZ!#;*_8y<tR@DDpHN=)SxEys80hL(v0S`pe5~SPX{{EjqdcICxs|X5sFfZ(v+brm8eV=s#1&E)S)hoXiO8D(u&r!p)H;0Oc%P+i{A91Fa7Ax00uIQ;f!D;;~38bCNhob%wQ(-n9l+hvW(@dU?uBV&jvQKjqU7UCxaNw5QZ{}(Trg%lbFmDrZS7!%waBzSj-ZZvWnHLVJ(~3%oet?i{0#DFZ<Zf0S<DE<DB3m=Qz&=E^>|Q+~6knxX%L~@{H%a;3e;P&j&v8jqm*6Cx<x95sq?-)12Wfm$=Lou5ydp+~F>dc+3-?@`~5I;Vqx|%oo1$i{Jd=?|)CX0Q^fpf)Sh$gd`l{i9kf65uF&sBp&feKths{oD`%a9qGwHMzWEe9ONVrfeAuTLJ^uUge4M@i9%Fj5t}%~B@u~9LQ+zZnlz*(6Pd|ER&tS>Jme)G`6)m_icy>rl%yQxsX#@lQJospq#pHYKtr0*oEEgC9qs8rN4n9S9`vLTg(*T&N>Q3Jl%*1tsX|q1QJXr{r4fy3LQ`7Nnl`kh6P@WoS9;N#KJ=v@{TaYOhB2HGjAR_+nZQJ*F`XIAWFGTbz(SU>oE5BO9qZY^Mz*n?9qeQfgBik5MlqT(jAas&nZi_NF`GHeWf6;6!ctbTnl-Ft6Pww>R(7$QJ?v#4`#Hctj&Yn5oa7wmxxhuPah)67<R15Vz(bz#oEN<09q;+TN51i$AN=GHhdIJgPH~zuoaGXixx!U$ahp5b<q?m0!c$)Hnm4@V6QB9QSAOxEKm6r?vQ`*9', 'block48-transition-map.npz': 'c-rllYfK#16~||Gmt6w}j32=@7?f6uD<_6&-Ij-0NQD&FBB`B5*qLJ3ky1}SAOkY2!HXAml}1)oRkX5FMp0~6O(PUM4a6fCvlN&*bxF&_RCPR**lX~U$Jp309&K5B+5Yz?Qd>0lYd>Ys1wQ<B?wxaf=W%CGrQNZOVVG4|PcyG<Wp2m6d^R!bnFeojqqo`9+~8?y@E&^Y0RF2xP@iAlc!XuvFaanMPw^Ljn0w)>$hl<x57(zET3XVVwm3IBHwxcex9-o+UpSxX{C;Mm^IK~Mt3Mq4@>#4I-`Co5@fXIqiIT$$osA3od)x0Qr#_cwa<Y2QwcdVj#?uyC{9v%zJkWLW$!|w%Lr>0B7PQWJ3q}^+?kbJEH{p(dwitg@S9>&ew###PE}B)lc&4tP#y?q++jRS8ukZYZuJT6@$7;Qk@u~2Q^Tma2olddFu0{QVelJ~ktEoORzsxlM_o{DXW#4MfIHJns4l#V4*?T$0&VIULU7fYPY8DClS*V<1c#fjF1y!q{@31S&>m_fO{ne_7>o>2gSWSaX_8jTa+p65+68$V^oT2EB5#491Yy^3aLkq=Ox#zL{@9(cz1D6Ka^cQ_Kg5Jg`!w$(*DfYUEJ)bH^9HRegobe}DU(Z6u^W7`f%o(D4PS9O8Wtf$`7byG2$-dj6vn#{NlJ~Z??#MLJr3(5n*h`Yke_Ee!ahmM=YgNvGU$Qx)hw>g5h`pVHzQv}rGm@F1%As=R#LD$-kiYNHeBb4aV*x(y&@9jP9A_-4@@`g)t>O$%fcJJVB`eoHR;gvQ=ywFYlU0_}B&$Q+eCNv0PP_5}43%-lVOQVZiu>DE8Y(=;>~%XOvsBRYY|26-b9yC>gve$PnZp@BhRUuiG*eWEpep5z6G6TsMGK8lViYF6dP&d+5hF(OPEvFq5#2#SH&wa7CWhk_!so6su4jhwIl~>`JvPlZO|dsd?76`tV!m23rzx?VB)X?K!ye?P7%g;zvTvO13uxbf4!Q8qT37mj=#n_20y>=UE``YlWZz0K3BROC-piEtxIyfFgEQ6z`Eo$u6847k0zX`_F203iwuwWVB(p-5J8Yum*`AWjqk`VeDD6nn0&8eb2lxU;^l##fQ9$dogmwUK_?aqa0mdB3+fVWA&r^lV#{>NR6fKI^bf*hdRtNumexM5U4<X%pgM6eUP-T??{R03uO)`HZ=u-f<$r{Op!}S6C@oUJ)YpUGKi0)gIVBRE7&EyQYt=A^H?^1L|Lg%(i=2k(UV3hoF>(fPw3)SGR21L69I#$U&Vtu~cL*#r2(yvtcKF&ue_N8~p`O-M!T|ktig`(EF{q4kFx@3L`d#vbQqS$*#?AbZvChWO1-#i7%^90JLIb(Z(FK4xAhBa)L0UIm&ku=?4B+0t4eVa2bsPZ`K5t>_vbz$3`Bn|;Zw7yezF_y=jw(16=Dx=QUI<#n$pob|KZA)N;1bCJ+3V=}&1)~r!LIbE*<q>pIgA(>T=%OgLm8fi~T4<b-nOBIj?5HrvV4GzAl9C6Vq^rE3%KK1>QY5pFqU$BP)q;)`^w_n~6UymkNHCK*BZjVwVBVqx^J8+pIziuweZlZgDP6CR?E5`c#;D@Y;*5F9`Id;i9fCdzdoIbmMX~oWv6l{e=%RU?@jj)?C(k&GxVbmM2qVTEoKa3`es<*=LdJs;n;6ToX5_D^Ms}qPr`oQ{an`#0ZIH~hf<D74?Z}u8>+(0jh(qWWXvrQ%+)O$7S-^%gA6_Gwm8zVbtM*u$Ld_b=kxXQPhZV#BqU2@_u%!rk7e)!RflCzUzmq+!{w@QB+07Xl0luG7i3&>-l_(Fn@&cMSt1Prqn)lH}<tRh$eS(1!r=wJktB(?}`8RS#HIl)pg$znE#1e2s9fBUg=$R{dM=0+xO5oU~%6SekhOWj^TzrwhaU4BwwHAF{(BGisea(1UqwtSzIB&gV_NsCv#a=tH_Y&-7XdzSwux=TRX_8#UoRJme3llmDOS7aiWJ~540NAYdjIxTApYu;7;?@svw{pfauD<!tDcE5!`-;1-xck3zHv(2r^T&ex2&;vbD6QlpjB<~0w_7Wj)v8=g0k3cv#tJy&G>oBtnG|DplKSRjXmP1p6yy293~L8}LL9skn{JlOEL9%Nv);7&Wa3RbZK8h*CddFEDW$ajGD0Y>;`Nvz8D+swA+(JUiViV^A;B(&BNReogwPVq$w5AzuK8vugw7B`UljD+Hf4F8WImt}dY=$Fi8kjHV}+cNNg-;O5Y+^t;6RU{ze^#sf;g}#z>lPBA>6urU$<_Z*NMG@h*`20!cE(^)>@ZIboZc3(?Ynd_)@HOFB9E>Do-Ve;X7n(w;U6qS!z!NVFUAp0ltt@SQ=WfQ=Xt>c4|+RyknG^<1YzgNunQj?G{w2GHYYwe*~bINb2wsE-iGA5-~F9L{|xVKWd6i@<u2T8zT{W8KXC9$_CCjNr{+FJn~wApGwoB*@8YqLH;0#7&3Y|OEOzj`7tHW22UlB-|Z4(MVv7e<TKoqYY8dJr;@ekE<x|GDY@;GYm=WKkD_u3!;@2V)9%0#f;{ftuLJp1&G#uK!)M8)n9dm!C}uk_m^-YC*%MOqwgG@3-(N%qcuU<EcVKj&e&|ZbebG`kPl_X)H>k>2a@3w~%i_5IC(C(zDp}h$=6^p!CO#@AJ}_=#{__Ob#BcgX$vu@x&V+)+&kykTP4Z##-`o9GFa', 'block56-transition-map.npz': 'c-qBUeN>cX7JmViH8a}UqGV<0w41x@;-sd8Ff*JwyVkl2oir9nrvh|^U`!ZTz?s3d;<)C*?j8?mE6}mh%%U(F7i7lrBp4}v1o$<eVxpjog*Y`Z%+C9Jo_pu)ANx0S%#U~8-@W&DfA`~gAZ1b5gjhvUeiHcgkdhVu-s67xHBI@MVlo%3H5V8QOvZeZd0j?U;7>VOd9(7?7KbQzD`m>B6!q(p2EQpcPJ5yb=IT5bbJlN~Qy*t9j(qoIQvIfsJ06<(e!}m5Q*`91=bn4;?d!THwq>;)`?|He)ZN^d|GfW;p@D(*nwB5ih9eK?yPkFD`%8N~A^C}UiDQS$&)<48yk~u{%e4Qy=m=-q7JJjh!{z$^D=polwF~t(pNTVEes9Ry9<IAIJI*!udX#49<8Yn#y(qW!Q0`!1S5x<oc|`|Yxu&8+_vZcJKNnlR%^bI3@08uk!&=WKb*@ojt}RdMoU~`<+R(HqyBmj<`bR3wA<MM!WqYS)?wF+fb7e_LTGZ~6Gm|r8Yx`dh$y^*W5V~h&hLRT5x?rmEi{{SI`uQot3+`=Qp42ftymFq3U9vGGGdE`7F{QGgYGP(?ZGWEnc<Z*;V0pm|<?e<Jdrv7zCmRA*mNs9QP=9}ld4RCjG4?HB*8_WMSnKwr&WTH#4=eTaE6vlCr%FBqw)v6DhO#{gcg5s{rbVU<PvW>{O$q#|!-dBo1-Z4^i(_&^mUU5tk5hzs&G6!~>c#L$GIO2GG`<J|wt#?!NixD69AOK*m<lsZcM31ec6$8S{gn+H_5|LUxO9hBiOGhKe`Oy#;p3tus%KhV+1|;SE0Q|e0<nap?FuP-Ckk(Oy_wV*;3Y>%ySKHmY){vNfrQmB2~(afdR|Rk_X|i}n+&WdiXNM0tAJ{V-Pz3PQI%73JECz=ux-q#9nrXEsL^;dNu3kwXVvzHO{$a$dlC+n!{3T%`S=P2d+Rjbegz3jms|`#$I<5E?NYq$nJ(X6!Qf4?fw-5s^(tA-ei@GDQOZ+vy_zRoff{YeE%`Vug1Bm#rM_M9mdaIk=+ywN=}5R~uB25LPKGRzsvkw$@X=gJYmtU{b30W-mV*#PrIcm#0`>!XaV4s5>I&uCEmO%4`=5}MY`FQBe0z}VX|M@uYXf*5C18XT;0J|akW)EPc$^u&ipN)wzE9-ax;)PAanPcw_FG2$2VVISh*gx*`<RTGT9voD1!_31Riv@$M20;b?ZJj{c2YQc4(Ahua|Yo&AsuoBOSly!{1Oz=79qp=8!KWfGLgDN%|sq0YJl~B4vMHVKoKWM;G8Irp!#L__E2r$qxOHJR5E24kz5N}<h1z`{QQb&@8=>%SqwMA&l^<aKBoQmqL-j7N+~O=IE>dI>~vAab7UBsSrS!7aQF)R{FeHhnMiW2LWgl7sfFa*7$<Ni5;zFobmZGU_U$&_P~Di$rIY~TJVN35k`2!^R`@=!(E~O<mJaFvOb+Q@f<vMs{!}_-`Gy?w3K3>NQFc<)xEK#_3}o#tFq1G<_TCz1rW#o*MH_IFx<)uH<?vJU7z+6%>E}WgUj@{qM<IK!lAo(s+CE5Te2gE(zXZ&D%5^;Dw<@4r%nAamV%D{S_4A;$pG0<rMRo(U9)<>&P#qVuM9oObA@y;#9JLMOU^;X~)b6ujd<u=)&CK{cVwi$)EivB4jHik!%tsa8kd)^AoQEm)ph2{Cmsti8$B*+>FUH5TDW2ze39Ii0)K>_HeAE=Wcq<WOM29K~m*O10z>iPi?G1RloK~kdIh+>}PI)rsXS@t2oky*G1^R9t_KqIemfbuc2Su-KLB4BgekSt1g=qCyIK9W|@mzYWe_3VPdlPbq-SANWX&6YKk#bzzwR;f8M}h=eJ}WqkT9EKL9xtX&a)CR^aXg+aNFa_ba#N4RY_eCQjnSO9^F;ArFm{_k0&Oeq*}Nruif!dDc)Uiw9l<%(G~)3pkiboFJ*W335AqJKlM`Z)V+f@_AlvR%Zo40#3hU-WMdWW6`)h{=2Oq^O{Z9I8nnV6}7og0BK>KTw%keR;=G-OrrxP?RgUoz^oc{q?`+!DqHfQZKNOu60$g}A#e!K-qjT5t}fJ%)w3he>xcH?7Q9g!2nh~&e`pg5cip#sXJL*1q9kQ)vi1aTKh;x}31mB`vN@OL5A`43zmuZmeJ)*7mSzq7yH2qyt;AcAt<%x!-J8+Qa&u4ujN65-q>BGS~HN8{{cxh!*D!O<p*30H^$i9D_@<UksvV^9Jkg+p9!9n|BNNWof)fVlK1H81b$^SGg`gSbXP;j2_?g*;z!@VEsOc2kcTV&NYI!`+~8geW}76y5}d8$n^jgOa0Xn8JsV@)l5-OcZ7?g)JamjZu9l!l~Jh<Hz^#H;r<7j2$`)hl=4)Drs;TTg<^OT8EX|LBnJ<b7=xu4vZr2eq!TAV#9I?=tqFQnj{g==xZQ}J}BHcTV^Vi2VEU<?m$w#of5rTN2zdF(A$rJ>)x_Jom;*@n42if?^(KgL2N!I*e-IUm>roZ!rYE9>DW@o$Ce(^UrXW0ZRv<3pTj(lFef9-8I;u{jJ^-(?Wol1`7+GDV3^RZLu_Z;37sWn836iqgn3T(QtvU=%#*yaFXRSVfnqY?@ar3u8h3IZuoi>44nTSc@(@F|ATKpIf!GYvVFRy|ww0>GV;EN6x240@Omf(F6%O};j&?exZDu;EpkgC5WwlAN4P1aWbWPtiq|zf(>FecG_92x<)QBcjUD+Ror3xaxmo4=POXvjFVrj|9X?0??@LhzZ7P54WEsemE29|7eKVjr@i@?%tm;;ipN|FtHn6+9#@|PgFib(EYl1D(YMv$B+Nv>g%ohYp_9P_+=lH@Ty=2`k_rk4a&ehUuIg~M9v51UyPRp@+1j8%(II^530_lQ;@rh|)W$j#^upogP|lIg-FsPa=FyB%aZUXWyKd2cu3!o?!a_QQ195UinIh~_9#Swx~~X7mx%kOrp${e9|%%)bz3G7ZMLoR>%uCJsZEBQi|md~O@aOUV^DvW3Fr7VRBGn595>M@jT>CUF-^D;bNN-Xg;cl4$%GVV(f`FA4n&qkDnA4PnNSVvll|O(4-N`mIT#Yq*;mczLgmW1Zo^@BI&S)<rry$~rp-ow<e1Xs{ZCIupe#nyiO(#(D91kQXN$P9}#BF#24ex1)tS3Ejl#JwUe>0ey(j`9{Y#40Jotos{(&ro#z3{GwIdqh@^w@~ffQ*vEOP!_AX39@Ti8T3!ONG`1crwSlENV(A>m;YJ)bu;izu!OwBj;BvG&4(MM~tp{@;M$u8A+X>yp=zTypiX0GnPq3kiS{Gx~O)M30ZH#iQUt%G>10g*NI@+l@z0b_$i)yTakk-*ei{^bv3~n5{N;+jcJ;55uKs+Cb(Mo5baWSs$?>f;1-=PH~i<!9%{d|Lu<`eIu%==dGo+=h5>J$U488MhmxaD<ZNI%`&vNV_v<7C{B$feFAwu;m2CRmJ==XCG8Y;h(m)}a_}k4l+oc+oImpBlprr?;GBwuXzb4e^*zj1g3f=NUZ%=*fRXsS~=E(Tza2<^p{qq4Q0g?+DOW1KmKIW-@EW2s&(_!}@?)>iJ^CQwyHK%xTf!ooMjBuc*{nKVvU|rJZ0YnOF*5aYi5x16cCY{i>7W(4b{iCjk94l{z2At@sX(_g$c;5_$uphl{Z;azN-cjBdlSH?{^WRS`?Md|!#fJ=ArUtL+U`+pob~B(>l#nYl%%HZ7{Hnzrb;J9$UP4T&QQ7OQEC<{J`+4e@-Ac%G(#H7>@Jx&j9;p`SPCqkO7&ti?w!;iHSF+P-Jrg`ZJqNv3I1GwG}u4>Tioa0e}&8Xnup;u>juj2q@rsayF<&UYQ<G75`vB(o3MVuG0JLT1Ffk!9u-rH&Tstst4@FnT-CZDOj^dFT{pbv3fuht}mJbQ7cZh*CEJeKVo+w*`IT`yWL>w^6AZ8Qq3b_lr`$LoM}WSTZC?OGCWNnMA4M*zY?=rOtDA4A=YK55SU%SZZRHG-9fYQm30#KgUsntXoSE#~I4HflFNsq7Q*yMCi?o9x0}}$N{1EFuGrqdJfQQh$R!VWJp;Q5*n-g@6QXU-~CsA5@4DV{3;7k|NsAgdLU)d#7XKWLIVFi75IHtjQY3#0v=R*X8'}



"""PTX-ISA matrix-fragment codec for QMMA m16n8k32 E4M3.

This is the first reusable primitive for static execution of the original
sm_120 kernels.  It implements NVIDIA PTX 9.3's published lane mappings for A,
B and f16 C/D fragments.  It does not yet claim tensor-core arithmetic/rounding
is bit-identical; fragment placement and byte/half packing are exact.
"""

import hashlib
from pathlib import Path

import numpy as np


HALF_TO_E4M3_TABLE_PATH = Path(__file__).resolve().parent / "tools" / "fp8_half_oracle.bin"
HALF_TO_E4M3_TABLE_SHA256 = "0212e2599adcd3301d3bad890a053b8b41e514049b9988db67e77c2e21e464ce"
HALF2_ORACLE_PATH = Path(__file__).resolve().parent / "tools" / "half2_oracle.bin"
HALF2_ORACLE_SHA256 = "ab7103b70165a098bdc56540368a6be3ee527780a7982063ae9c7ddbe6a431cb"


def load_half_to_e4m3_table(path: Path = HALF_TO_E4M3_TABLE_PATH) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) != 65536 or hashlib.sha256(raw).hexdigest() != HALF_TO_E4M3_TABLE_SHA256:
        raise ValueError(f"invalid FP16->E4M3 oracle table: {path}")
    return np.frombuffer(raw, dtype=np.uint8)


def encode_half_e4m3(value: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    """Convert FP16 values to E4M3 bytes via exhaustive local hardware oracle."""
    source = np.asarray(value, dtype=np.float16)
    bits = source.view(np.uint16)
    if table is None:
        table = load_half_to_e4m3_table()
    return np.asarray(table, dtype=np.uint8)[bits]


def quantize_half_e4m3(value: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    return decode_e4m3(encode_half_e4m3(value, table))


def decode_e4m3(value: np.ndarray) -> np.ndarray:
    byte = np.asarray(value, dtype=np.uint8)
    sign = np.where(byte & 0x80, -1.0, 1.0)
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 7
    result = np.where(
        exponent == 0,
        sign * (mantissa / 8.0) * (2.0**-6),
        sign * (1.0 + mantissa / 8.0) * np.exp2(exponent.astype(np.float32) - 7.0),
    )
    result = np.where((exponent == 0xF) & (mantissa == 7), np.nan, result)
    return result.astype(np.float32)


def unpack_a_bytes(lane_bytes: np.ndarray) -> np.ndarray:
    """Decode [32 lanes,16 E4M3 bytes] into logical A[16,32]."""
    source = np.asarray(lane_bytes, dtype=np.uint8)
    if source.shape != (32, 16):
        raise ValueError(source.shape)
    output = np.empty((16, 32), dtype=np.uint8)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for index in range(16):
            row = group if index < 4 or 8 <= index < 12 else group + 8
            col = thread * 4 + (index & 3) + (16 if index >= 8 else 0)
            output[row, col] = source[lane, index]
    return output


def pack_a_bytes(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.uint8)
    if source.shape != (16, 32):
        raise ValueError(source.shape)
    output = np.empty((32, 16), dtype=np.uint8)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for index in range(16):
            row = group if index < 4 or 8 <= index < 12 else group + 8
            col = thread * 4 + (index & 3) + (16 if index >= 8 else 0)
            output[lane, index] = source[row, col]
    return output


def unpack_b_bytes(lane_bytes: np.ndarray) -> np.ndarray:
    """Decode [32 lanes,8 E4M3 bytes] into logical B[32,8]."""
    source = np.asarray(lane_bytes, dtype=np.uint8)
    if source.shape != (32, 8):
        raise ValueError(source.shape)
    output = np.empty((32, 8), dtype=np.uint8)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for index in range(8):
            row = thread * 4 + (index & 3) + (16 if index >= 4 else 0)
            output[row, group] = source[lane, index]
    return output


def pack_b_bytes(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.uint8)
    if source.shape != (32, 8):
        raise ValueError(source.shape)
    output = np.empty((32, 8), dtype=np.uint8)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for index in range(8):
            row = thread * 4 + (index & 3) + (16 if index >= 4 else 0)
            output[lane, index] = source[row, group]
    return output


def unpack_half2(words: np.ndarray) -> np.ndarray:
    source = np.asarray(words, dtype=np.uint32).copy()
    return source.view(np.uint16).reshape(source.shape + (2,)).view(np.float16)


def pack_half2(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float16)
    if source.shape[-1] != 2:
        raise ValueError(source.shape)
    return source.view(np.uint16).reshape(-1, 2).reshape(source.shape[:-1] + (2,)).view(np.uint32).reshape(source.shape[:-1])


def hadd2(a_words: np.ndarray, b_words: np.ndarray) -> np.ndarray:
    a, b = unpack_half2(a_words), unpack_half2(b_words)
    with np.errstate(all="ignore"):
        result = (a + b).astype(np.float16)
    return pack_half2(result)


def hmul2(a_words: np.ndarray, b_words: np.ndarray) -> np.ndarray:
    a, b = unpack_half2(a_words), unpack_half2(b_words)
    with np.errstate(all="ignore"):
        result = (a * b).astype(np.float16)
    return pack_half2(result)


def hmnmx2(a_words: np.ndarray, b_words: np.ndarray, *, minimum: bool) -> np.ndarray:
    a, b = unpack_half2(a_words), unpack_half2(b_words)
    result = np.minimum(a, b) if minimum else np.maximum(a, b)
    return pack_half2(result.astype(np.float16))


def hfma2(a_words: np.ndarray, b_words: np.ndarray, c_words: np.ndarray) -> np.ndarray:
    a, b, c = unpack_half2(a_words), unpack_half2(b_words), unpack_half2(c_words)
    # Binary16 product is exact in binary64; one final cast reproduces fused RN.
    with np.errstate(all="ignore"):
        result = (
            a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)
        ).astype(np.float16)
    return pack_half2(result)


def f2fp_e4m3_f16_merge_c(
    source_words: np.ndarray,
    merge_words: np.ndarray | int = 0,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """SASS F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C model.

    Two FP16 lanes become the low two E4M3 bytes; merge_words.low16 supplies the
    high two bytes.  This matches the packing chains immediately before STG.
    """
    encoded = encode_half_e4m3(unpack_half2(source_words), table).astype(np.uint32)
    low = encoded[..., 0] | (encoded[..., 1] << np.uint32(8))
    merge = np.asarray(merge_words, dtype=np.uint32)
    return low | ((merge & np.uint32(0xFFFF)) << np.uint32(16))


def f2fp_f16_e4m3_unpack_b(
    source_words: np.ndarray,
    *,
    high: bool = False,
) -> np.ndarray:
    """SASS F2FP.F16.E4M3.UNPACK_B model for low/high byte pair."""
    source = np.asarray(source_words, dtype=np.uint32)
    shift = np.uint32(16 if high else 0)
    pair = (source >> shift) & np.uint32(0xFFFF)
    bytes_ = pair.copy().view(np.uint8).reshape(pair.shape + (4,))[..., :2]
    values = decode_e4m3(bytes_).astype(np.float16)
    return pack_half2(values)


def prmt(a_words: np.ndarray, b_words: np.ndarray, selector: int) -> np.ndarray:
    a = np.asarray(a_words, dtype=np.uint32)
    b = np.asarray(b_words, dtype=np.uint32)
    a, b = np.broadcast_arrays(a, b)
    a_bytes = a.copy().view(np.uint8).reshape(a.shape + (4,))
    b_bytes = b.copy().view(np.uint8).reshape(b.shape + (4,))
    source = np.concatenate((a_bytes, b_bytes), axis=-1)
    output = np.empty(a.shape + (4,), dtype=np.uint8)
    for byte in range(4):
        index = (selector >> (byte * 4)) & 7
        output[..., byte] = source[..., index]
    return output.reshape(-1, 4).copy().view(np.uint32).reshape(a.shape)


def shfl_bfly(words: np.ndarray, mask: int, clamp: int = 0x1F) -> np.ndarray:
    source = np.asarray(words, dtype=np.uint32)
    if source.shape != (32,):
        raise ValueError(source.shape)
    lanes = np.arange(32, dtype=np.uint32)
    targets = lanes ^ np.uint32(mask)
    valid = targets <= np.uint32(clamp)
    return np.where(valid, source[targets & np.uint32(31)], source).astype(np.uint32)


def shfl_idx(words: np.ndarray, indices: np.ndarray, clamp: int = 0x1F) -> np.ndarray:
    source = np.asarray(words, dtype=np.uint32)
    target = np.asarray(indices, dtype=np.uint32)
    if source.shape != (32,) or target.shape != (32,):
        raise ValueError((source.shape, target.shape))
    valid = target <= np.uint32(clamp)
    return np.where(valid, source[target & np.uint32(31)], source).astype(np.uint32)


def unpack_f16_accumulator(lane_words: np.ndarray) -> np.ndarray:
    """Decode [32 lanes,2 uint32 f16x2 words] into logical C/D[16,8]."""
    words = np.asarray(lane_words, dtype=np.uint32)
    if words.shape != (32, 2):
        raise ValueError(words.shape)
    halves = words.view(np.uint16).reshape(32, 4)
    output = np.empty((16, 8), dtype=np.uint16)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for index in range(4):
            row = group if index < 2 else group + 8
            col = thread * 2 + (index & 1)
            output[row, col] = halves[lane, index]
    return output.view(np.float16)


def pack_f16_accumulator(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float16)
    if source.shape != (16, 8):
        raise ValueError(source.shape)
    bits = source.view(np.uint16)
    halves = np.empty((32, 4), dtype=np.uint16)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for index in range(4):
            row = group if index < 2 else group + 8
            col = thread * 2 + (index & 1)
            halves[lane, index] = bits[row, col]
    return halves.reshape(32, 2, 2).view(np.uint32).reshape(32, 2)


def qmma_reference(
    a_lane_bytes: np.ndarray,
    b_lane_bytes: np.ndarray,
    c_lane_words: np.ndarray | None = None,
) -> np.ndarray:
    """Mathematical QMMA reference; arithmetic rounding is not yet certified."""
    a = decode_e4m3(unpack_a_bytes(a_lane_bytes))
    b = decode_e4m3(unpack_b_bytes(b_lane_bytes))
    if c_lane_words is None:
        c = np.zeros((16, 8), dtype=np.float32)
    else:
        c = unpack_f16_accumulator(c_lane_words).astype(np.float32)
    with np.errstate(all="ignore"):
        result = (a @ b + c).astype(np.float16)
    return pack_f16_accumulator(result)


def _self_test() -> None:
    generator = np.random.default_rng(0x297)
    # Avoid E4M3 NaN payload 0x7f/0xff in round-trip fixtures.
    a = generator.integers(0, 0x7F, size=(16, 32), dtype=np.uint8)
    b = generator.integers(0, 0x7F, size=(32, 8), dtype=np.uint8)
    c = generator.normal(size=(16, 8)).astype(np.float16)
    assert np.array_equal(unpack_a_bytes(pack_a_bytes(a)), a)
    assert np.array_equal(unpack_b_bytes(pack_b_bytes(b)), b)
    assert np.array_equal(unpack_f16_accumulator(pack_f16_accumulator(c)).view(np.uint16), c.view(np.uint16))
    d = qmma_reference(pack_a_bytes(a), pack_b_bytes(b), pack_f16_accumulator(c))
    assert d.shape == (32, 2)
    table = load_half_to_e4m3_table()
    all_half = np.arange(65536, dtype=np.uint16).view(np.float16)
    assert np.array_equal(encode_half_e4m3(all_half, table), table)

    raw = HALF2_ORACLE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HALF2_ORACLE_SHA256:
        raise ValueError("invalid half2 hardware oracle")
    records = np.frombuffer(raw, dtype=np.dtype([
        ("a", "<u4"), ("b", "<u4"), ("c", "<u4"),
        ("add", "<u4"), ("mul", "<u4"), ("fma", "<u4"),
    ]))
    values = [unpack_half2(records[name]) for name in ("a", "b", "c")]
    finite = np.isfinite(values[0]) & np.isfinite(values[1]) & np.isfinite(values[2])
    for expected, actual in (
        (records["add"], hadd2(records["a"], records["b"])),
        (records["mul"], hmul2(records["a"], records["b"])),
        (records["fma"], hfma2(records["a"], records["b"], records["c"])),
    ):
        expected_bits = expected.copy().view(np.uint16).reshape(-1, 2)
        actual_bits = actual.copy().view(np.uint16).reshape(-1, 2)
        if not np.array_equal(expected_bits[finite], actual_bits[finite]):
            raise AssertionError("half2 hardware mismatch")

    packed_fp8 = f2fp_e4m3_f16_merge_c(records["a"], records["b"], table)
    decoded_low = f2fp_f16_e4m3_unpack_b(packed_fp8)
    decoded_high = f2fp_f16_e4m3_unpack_b(packed_fp8, high=True)
    expected_low = pack_half2(quantize_half_e4m3(unpack_half2(records["a"]), table).astype(np.float16))
    expected_high = pack_half2(
        decode_e4m3(records["b"].copy().view(np.uint8).reshape(-1, 4)[:, :2]).astype(np.float16)
    )
    assert np.array_equal(decoded_low, expected_low)
    assert np.array_equal(decoded_high, expected_high)

    a_word = np.array([0x03020100], dtype=np.uint32)
    b_word = np.array([0x07060504], dtype=np.uint32)
    assert int(prmt(a_word, b_word, 0x5410)[0]) == 0x05040100
    assert int(prmt(a_word, b_word, 0x5432)[0]) == 0x05040302
    lanes = np.arange(32, dtype=np.uint32)
    assert np.array_equal(shfl_bfly(lanes, 2), lanes ^ 2)
    indices = (31 - lanes).astype(np.uint32)
    assert np.array_equal(shfl_idx(lanes, indices), indices)

    print("PTX m16n8k32 fragment round-trips: exact")
    print("FP16->E4M3 hardware table:", len(table), "entries exact")
    print("half2 add/mul/fma hardware samples:", int(finite.sum()), "lanes exact")
    print("F2FP pack/unpack byte chains:", len(records), "words exact")
    print("QMMA reference output words:", d.shape)


if __name__ == "__main__":
    _self_test()



def qmma_a_physical_map() -> np.ndarray:
    result = np.empty((64, 32), dtype=np.int32)
    for tile_group in range(4):
        for lane in range(32):
            group, thread = lane >> 2, lane & 3
            for index in range(16):
                row = group if index < 4 or 8 <= index < 12 else group + 8
                col = thread * 4 + (index & 3) + (16 if index >= 8 else 0)
                result[tile_group * 16 + row, col] = tile_group * 512 + lane * 16 + index
    return result


def _decode_subtile(raw: bytes, offset: int) -> tuple[np.ndarray, np.ndarray]:
    tile = np.frombuffer(raw[offset:offset + 512], dtype=np.uint8).reshape(32, 16)
    return unpack_b_bytes(tile[:, :8]), unpack_b_bytes(tile[:, 8:])


def _n_concat(raw: bytes, offsets: list[int]) -> np.ndarray:
    return np.concatenate([fragment for offset in offsets for fragment in _decode_subtile(raw, offset)], axis=1)


def _k_concat(raw: bytes, groups: list[list[int]]) -> np.ndarray:
    return np.concatenate([_n_concat(raw, offsets) for offsets in groups], axis=0)


def _write_1h_matrices(root: Path, block: int, raw: bytes) -> None:
    shift = 0x400 if block == 0 else (0x70 if block == 70 else 0)
    w1_stream0 = decode_e4m3(_n_concat(raw, [0x000, 0x200, 0x800, 0xA00]).T)
    w1_stream1 = decode_e4m3(_n_concat(raw, [0x400, 0x600, 0xC00, 0xE00]).T)
    w2_stream0 = decode_e4m3(_k_concat(raw, [[0x1000, 0x1200], [0x1800, 0x1A00]]).T)
    w2_stream1 = decode_e4m3(_k_concat(raw, [[0x1400, 0x1600], [0x1C00, 0x1E00]]).T)
    offsets = {
        "qe": 0x2060 + shift, "qo": 0x2260 + shift,
        "ke": 0x2460 + shift, "ko": 0x2660 + shift,
        "ve": 0x2860 + shift, "vo": 0x2A60 + shift,
    }
    values = {name: decode_e4m3(_n_concat(raw, [offset]).T) for name, offset in offsets.items()}
    values["projection"] = decode_e4m3(_n_concat(raw, [0x4C70 + shift, 0x4E70 + shift]).T)
    np.savez(
        root / f"block{block}-qmma-matrices.npz",
        w1_stream0=w1_stream0, w1_stream1=w1_stream1,
        w2_stream0=w2_stream0, w2_stream1=w2_stream1,
        **values,
    )


def _write_2h_matrices(root: Path, block: int, raw: bytes) -> None:
    def matrix16(low: int) -> np.ndarray:
        return decode_e4m3(_k_concat(raw, [[low], [low + 0x1800]]).T)

    q_heads, k_heads, v_heads, projection_heads = [], [], [], []
    for head in range(2):
        shift = 0x0C00 * head
        chunks = [matrix16(low + shift) for low in (
            0x70A0, 0x72A0, 0x74A0, 0x76A0, 0x78A0, 0x7AA0
        )]
        q_heads.append(np.concatenate(chunks[0:2], axis=0))
        k_heads.append(np.concatenate(chunks[2:4], axis=0))
        v_heads.append(np.concatenate(chunks[4:6], axis=0))
        base = 0xE0B0 + 0x0400 * head
        projection_heads.append(decode_e4m3(_k_concat(raw, [
            [base, base + 0x200], [base + 0x800, base + 0xA00]
        ]).T))
    np.savez(
        root / f"block{block}-2h-direct-matrices.npz",
        q_weight=np.stack(q_heads), k_weight=np.stack(k_heads),
        v_weight=np.stack(v_heads), projection_weight=np.stack(projection_heads),
        scale=np.frombuffer(raw, dtype="<f4", count=2, offset=0xE0A0).copy(),
    )


def _parse_original_weights_bin(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    blob = path.read_bytes()
    if len(blob) < 8 or struct.unpack_from("<Q", blob, 0)[0] != len(blob):
        raise ValueError(f"{path} is not a complete DLSSNR WEIGHTS_HT export")
    records: list[dict[str, object]] = []
    cursor = 8
    seen: set[str] = set()
    while cursor < len(blob):
        record_offset = cursor
        if cursor + 8 > len(blob):
            raise ValueError("truncated record name length")
        name_length = struct.unpack_from("<Q", blob, cursor)[0]
        cursor += 8
        if not 1 <= name_length <= 4096 or cursor + name_length + 8 > len(blob):
            raise ValueError(f"invalid name framing at 0x{record_offset:x}")
        name = blob[cursor:cursor + name_length].decode("ascii")
        cursor += name_length
        body_span = struct.unpack_from("<Q", blob, cursor)[0]
        body_offset = cursor + 8
        cursor = body_offset + body_span
        if cursor > len(blob) or body_span < 40:
            raise ValueError(f"record {name!r} overruns the archive")
        total_size, data_size = struct.unpack_from("<QQ", blob, body_offset)
        device = struct.unpack_from("<I", blob, body_offset + 16)[0]
        if total_size != body_span or data_size & 1:
            raise ValueError(f"invalid body for {name!r}")
        data_offset = body_offset + 20
        trailer = data_offset + data_size
        if trailer + 16 > cursor:
            raise ValueError(f"truncated trailer for {name!r}")
        f1, f2 = struct.unpack_from("<II", blob, trailer)
        shape_count = struct.unpack_from("<Q", blob, trailer + 8)[0]
        shape_offset = trailer + 16
        shape_end = shape_offset + 4 * shape_count
        if shape_end != cursor:
            raise ValueError(f"invalid shape trailer for {name!r}")
        shape = list(struct.unpack_from(f"<{shape_count}I", blob, shape_offset))
        if name in seen:
            raise ValueError(f"duplicate weight record {name!r}")
        seen.add(name)
        records.append({
            "index": len(records), "name": name, "payload_offset": data_offset,
            "payload_size": data_size, "element_count": data_size // 2,
            "device": device, "f1": f1, "f2": f2, "shape": shape,
        })
    if cursor != len(blob) or len(records) != 153:
        raise ValueError(f"expected 153 records, parsed {len(records)}")
    return blob, records


def _materialize_original_weights(path: Path, target: Path) -> None:
    blob, records = _parse_original_weights_bin(path)
    weights = target / "weights"
    weights.mkdir(parents=True)
    manifest = []
    raw_by_name: dict[str, bytes] = {}
    for item in records:
        name = str(item["name"])
        begin = int(item["payload_offset"])
        size = int(item["payload_size"])
        raw = blob[begin:begin + size]
        raw_by_name[name] = raw
        safe = name.replace(".", "_")
        filename = f"{int(item['index']):03d}_{safe}.npy"
        np.save(weights / filename, np.frombuffer(raw, dtype="<f2").copy())
        manifest.append({
            "index": int(item["index"]), "name": name, "file": filename,
            "element_count": size // 2, "data_bytes": size,
            "f1": int(item["f1"]), "f2": int(item["f2"]),
            "device": int(item["device"]), "dtype": "fp16",
        })
    (weights / "manifest.json").write_text(
        json.dumps({"archive_size": len(blob), "count": len(manifest), "tensors": manifest}),
        encoding="utf-8",
    )
    for name, encoded in _EMBEDDED_STATIC_ASSETS.items():
        (target / name).write_bytes(zlib.decompress(base64.b85decode(encoded.encode("ascii"))))
    for block in (0, 1, 2, 3, 4, 67, 68, 69, 70):
        _write_1h_matrices(target, block, raw_by_name[f"block{block}.layer0.layer"])
    # block66 interleaves its 64->32 transition between the ordinary FFN and
    # attention regions; rebuild the ordinary 1H record before extraction.
    fused66 = raw_by_name["block66.layer0.layer"]
    body66 = bytearray(0x50C0)
    body66[0x0000:0x2000] = fused66[0x0000:0x2000]
    body66[0x2010:0x2050] = fused66[0x2810:0x2850]
    body66[0x2060:0x5070] = fused66[0x28A0:0x58B0]
    body66[0x5070:0x50B0] = fused66[0x58B0:0x58F0]
    _write_1h_matrices(target, 66, bytes(body66))
    for block in (5, 6, 7, 8):
        _write_2h_matrices(target, block, raw_by_name[f"block{block}.layer0.layer"])



"""Canonical execution of the physical two-stream 1H/32 FFN.

This uses actual E4M3 archive fragments, the recovered canonical/physical tile
permutation and the symbolic W1-D -> W2-A hidden routing. PyTorch/NumPy matmul
is intentional because the target is algorithm equivalence, not Blackwell
instruction matching.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def f16_fma(a: np.ndarray, b: np.ndarray | float, c: np.ndarray | float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (
            a.astype(np.float64) * np.asarray(b, dtype=np.float16).astype(np.float64)
            + np.asarray(c, dtype=np.float16).astype(np.float64)
        ).astype(np.float16)


def fast_activation_f16(value: np.ndarray) -> np.ndarray:
    original = np.asarray(value, dtype=np.float16)
    clipped = np.clip(original, np.float16(-4.0), np.float16(4.0))
    gate = f16_fma(np.abs(clipped), np.float16(-0.055908203125), np.float16(0.447265625))
    gate = f16_fma(clipped, gate, np.float16(0.89453125))
    with np.errstate(all="ignore"):
        return (original * gate).astype(np.float16)


class Physical1HFFN:
    def __init__(
        self,
        root: Path = ROOT,
        block: int = 1,
        *,
        values: np.ndarray | None = None,
    ) -> None:
        if block not in (0, 1, 2, 3, 4, 66, 67, 68, 69, 70):
            raise NotImplementedError(block)
        matrix_path = root / f"block{block}-qmma-matrices.npz"
        if not matrix_path.exists():
            raise FileNotFoundError(f"run extract_block1_qmma_fragments.py first: {matrix_path}")
        matrices = np.load(matrix_path)
        self.w1 = [matrices[f"w1_stream{stream}"].astype(np.float32) for stream in range(2)]
        self.w2 = [matrices[f"w2_stream{stream}"].astype(np.float32) for stream in range(2)]
        routing = json.loads((root / "block1-ffn-hidden-routing.json").read_text())
        route = routing["hidden_n_to_global_w2_k"]
        self.hidden_to_w2_k = np.array([route[str(index)][0] for index in range(64)], dtype=np.int32)
        output_route = routing["w2_output_n_to_qkv_k"]
        self.output_to_qkv_k = np.array(
            [output_route[str(index)][0] for index in range(32)], dtype=np.int32
        )
        if len(np.unique(self.hidden_to_w2_k)) != 64:
            raise AssertionError("hidden routing is not a permutation")
        if len(np.unique(self.output_to_qkv_k)) != 32:
            raise AssertionError("W2 output routing is not a permutation")
        if values is None:
            weight_matches = sorted((root / "weights").glob(f"*_block{block}_layer0_layer.npy"))
            if len(weight_matches) != 1:
                raise FileNotFoundError((block, weight_matches))
            raw = np.load(weight_matches[0]).astype("<f2", copy=False)
        else:
            raw = np.asarray(values).astype("<f2", copy=False)
        skip_offset = 0x2410 if block == 0 else 0x2010
        self.skip_physical = np.frombuffer(
            raw.tobytes(), dtype="<f2", count=32, offset=skip_offset
        ).astype(np.float16)
        permutation = np.fromfile(root / "tinlayout-1h32-permutation.i32", dtype="<i4").reshape(64, 32)
        # Derive the same separable qmma-row/column -> canonical maps used by
        # recover_1h32_permutation.py.
        inverse = np.empty((2048, 2), dtype=np.int32)
        for token in range(64):
            for channel in range(32):
                inverse[permutation[token, channel]] = (token, channel)
        mapping = inverse[qmma_a_physical_map()]
        self.token_permutation = mapping[:, 0, 0].copy()
        self.channel_permutation = mapping[0, :, 1].copy()
        # +0x2010 is stored in W2 output-N order. The residual source is in
        # QKV-K order, so qkv[k] consumes storage[inverse_route[k]]. A controlled
        # skip-only SM89 run recovers this permutation with zero element error.
        self.skip = self.skip_physical[np.argsort(self.output_to_qkv_k)]
        if [matrix.shape for matrix in self.w1] != [(64, 32), (64, 32)]:
            raise AssertionError([matrix.shape for matrix in self.w1])
        if [matrix.shape for matrix in self.w2] != [(32, 64), (32, 64)]:
            raise AssertionError([matrix.shape for matrix in self.w2])

    def group(self, input_e4m3: np.ndarray) -> np.ndarray:
        encoded = np.asarray(input_e4m3, dtype=np.uint8)
        if encoded.shape != (16, 32):
            raise ValueError(encoded.shape)
        source = decode_e4m3(encoded).astype(np.float16)
        # Static provenance closes the pairing: W2 component0 consumes only W1
        # stream0; component1 consumes only stream1.  Each branch has its own
        # activation/quantization, then all four (component,K-slice) QMMA terms
        # accumulate with the residual before QKV.
        hidden_w2: list[np.ndarray] = []
        for component in range(2):
            with np.errstate(all="ignore"):
                hidden = (
                    source.astype(np.float32) @ self.w1[component].T
                ).astype(np.float16)
            hidden_q = quantize_half_e4m3(fast_activation_f16(hidden)).astype(np.float32)
            reordered = np.empty_like(hidden_q)
            reordered[:, self.hidden_to_w2_k] = hidden_q
            hidden_w2.append(reordered)
        residual_qkv = (source * self.skip).astype(np.float16)
        residual = residual_qkv[:, self.output_to_qkv_k]
        with np.errstate(all="ignore"):
            output_n = (hidden_w2[0][:, :32] @ self.w2[0][:, :32].T).astype(np.float16)
            for component, begin in ((1, 0), (0, 32), (1, 32)):
                contribution = hidden_w2[component][:, begin : begin + 32] @ self.w2[component][:, begin : begin + 32].T
                output_n = (output_n.astype(np.float32) + contribution).astype(np.float16)
            output_n = (output_n.astype(np.float32) + residual.astype(np.float32)).astype(np.float16)
        output_qkv = np.empty_like(output_n)
        output_qkv[:, self.output_to_qkv_k] = output_n
        return encode_half_e4m3(output_qkv)

    def canonical_to_qmma(self, source: np.ndarray) -> np.ndarray:
        value = np.asarray(source, dtype=np.uint8)
        if value.shape != (64, 32):
            raise ValueError(value.shape)
        output = np.empty((64, 32), dtype=np.uint8)
        for group in range(4):
            tokens = self.token_permutation[group * 16 : (group + 1) * 16]
            output[group * 16 : (group + 1) * 16] = value[tokens][:, self.channel_permutation]
        return output

    def qmma_to_canonical(self, source: np.ndarray) -> np.ndarray:
        value = np.asarray(source, dtype=np.uint8)
        if value.shape != (64, 32):
            raise ValueError(value.shape)
        output = np.empty((64, 32), dtype=np.uint8)
        for group in range(4):
            tokens = self.token_permutation[group * 16 : (group + 1) * 16]
            output[np.ix_(tokens, self.channel_permutation)] = value[
                group * 16 : (group + 1) * 16
            ]
        return output

    def qmma_groups(self, input_e4m3: np.ndarray) -> np.ndarray:
        """Process input already arranged as four PTX A-fragment matrices."""
        source = np.asarray(input_e4m3, dtype=np.uint8)
        if source.shape != (4, 16, 32):
            raise ValueError(source.shape)
        return np.stack([self.group(source[group]) for group in range(4)])

    def tile(self, input_e4m3: np.ndarray) -> np.ndarray:
        """Process one canonical row-major 8x8x32 E4M3 tile."""
        source = np.asarray(input_e4m3, dtype=np.uint8)
        if source.shape != (64, 32):
            raise ValueError(source.shape)
        qmma_input = self.canonical_to_qmma(source).reshape(4, 16, 32)
        qmma_output = self.qmma_groups(qmma_input).reshape(64, 32)
        return self.qmma_to_canonical(qmma_output)


"""Phase-specific 1H/32 attention grouping recovered from the SM89 launch ABI.

This is intentionally independent from ``physical_1h_block.py`` so the recovered
layout can be validated before integration.  The native kernel receives a physical-cell origin in parameter q4 /
c[0][0x180].  A 512-byte cell is not a canonical 4x4 patch: TinLayout's
``p=64*(token//2)+...`` makes it one 2x8 token stripe, and the four stripes are
placed in TL, BL, BR, TR cell order.  Native attention pairs 2x2 such cells.
This is why an ordinary HWC roll is wrong.  Missing boundary cells are
zero-filled and only valid query slots are published.  The resulting grids are
40x32, 41x33, 41x32 and 40x33 for the four phases at 320x256.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn



@dataclass(frozen=True)
class Phase1H:
    block: int
    offset_y: int
    offset_x: int

    def grid(self, height: int, width: int) -> tuple[int, int]:
        """Return native (grid_y, grid_x) for this phase."""
        return ((height - self.offset_y + 7) // 8, (width - self.offset_x + 7) // 8)


# q4 is stored as (x, y); tuples here follow tensor (y, x) convention.
ENCODER_1H_PHASES = {
    1: Phase1H(1, 0, 0),
    2: Phase1H(2, -4, -4),
    3: Phase1H(3, 0, -4),
    4: Phase1H(4, -4, 0),
    66: Phase1H(66, 0, 0),
    67: Phase1H(67, -4, -4),
    68: Phase1H(68, 0, -4),
    69: Phase1H(69, -4, 0),
    70: Phase1H(70, -4, -4),
}


def phase_group_indices(
    height: int,
    width: int,
    phase: Phase1H,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(members, valid, slots)`` for native phase CTAs.

    ``members[g,s]`` is a flattened canonical HWC pixel index.  Invalid entries
    are clamped to zero and must be interpreted through ``valid[g,s]``.
    ``slots`` is the native 8x8 row-major slot (0..63); it is deliberately not
    compacted at boundaries because the learned 64x64 bias is slot-relative.
    """
    if height <= 0 or width <= 0:
        raise ValueError((height, width))
    if phase.offset_y % 4 or phase.offset_x % 4:
        raise ValueError("1H phase offsets must be multiples of one 4x4 physical cell")
    gy, gx = phase.grid(height, width)
    y = torch.arange(height, device=device, dtype=torch.int64)[:, None].expand(height, width)
    x = torch.arange(width, device=device, dtype=torch.int64)[None, :].expand(height, width)

    # decode_raw's 2048-byte tile consists of four 512-byte cells.  Because
    # p=64*(token//2)+..., those cells hold 2x8 token stripes, physically placed
    # in TL, BL, BR, TR order (not four canonical 4x4 quadrants).
    stripe = (y & 7) >> 1
    tile_y, tile_x = y >> 3, x >> 3
    cell_dy = ((stripe == 1) | (stripe == 2)).to(torch.int64)
    cell_dx = ((stripe == 2) | (stripe == 3)).to(torch.int64)
    cell_y = 2 * tile_y + cell_dy
    cell_x = 2 * tile_x + cell_dx

    origin_cell_y, origin_cell_x = phase.offset_y // 4, phase.offset_x // 4
    group_y = torch.div(cell_y - origin_cell_y, 2, rounding_mode="floor")
    group_x = torch.div(cell_x - origin_cell_x, 2, rounding_mode="floor")
    group = group_y * gx + group_x

    local_cell_y = (cell_y - origin_cell_y) & 1
    local_cell_x = (cell_x - origin_cell_x) & 1
    # Invert TL,BL,BR,TR to the virtual row-major 8x8 slot consumed by the
    # already recovered phase-0 token permutation and learned bias matrix.
    stripe_lut = torch.tensor((0, 3, 1, 2), device=device, dtype=torch.int64)
    local_stripe = stripe_lut[2 * local_cell_y + local_cell_x]
    slot = (2 * local_stripe + (y & 1)) * 8 + (x & 7)

    members = torch.zeros((gy * gx, 64), device=device, dtype=torch.int64)
    valid = torch.zeros((gy * gx, 64), device=device, dtype=torch.bool)
    flat = (y * width + x).reshape(-1)
    gf, sf = group.reshape(-1), slot.reshape(-1)
    members[gf, sf] = flat
    valid[gf, sf] = True
    slots = torch.arange(64, device=device, dtype=torch.int64).expand(gy * gx, -1)
    return members, valid, slots




"""Physical 1H/32 algorithm recovered from the original SASS.

The FFN routing, Qe/Qo/Ke/Ko/Ve/Vo topology, bias/scale, projection route and
residual gates use directly decoded archive components. NumPy/PyTorch matrix
arithmetic intentionally replaces instruction-level QMMA accumulation.
"""

from pathlib import Path

import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent


class Physical1HAttention:
    def __init__(
        self,
        root: Path = ROOT,
        block: int = 1,
        *,
        values: np.ndarray | None = None,
    ) -> None:
        if block not in (0, 1, 2, 3, 4, 66, 67, 68, 69, 70):
            raise NotImplementedError(block)
        self.ffn_layout = Physical1HFFN(root, block, values=values)
        matrices = np.load(root / f"block{block}-qmma-matrices.npz")
        self.qe = matrices["qe"].astype(np.float32)
        self.qo = matrices["qo"].astype(np.float32)
        self.ke = matrices["ke"].astype(np.float32)
        self.ko = matrices["ko"].astype(np.float32)
        self.ve = matrices["ve"].astype(np.float32)
        self.vo = matrices["vo"].astype(np.float32)
        self.projection = matrices["projection"].astype(np.float32)
        for name in ("qe", "qo", "ke", "ko", "ve", "vo"):
            if getattr(self, name).shape != (16, 32):
                raise AssertionError((name, getattr(self, name).shape))
        if self.projection.shape != (32, 32):
            raise AssertionError(self.projection.shape)

        if values is None:
            weight_matches = sorted((root / "weights").glob(f"*_block{block}_layer0_layer.npy"))
            if len(weight_matches) != 1:
                raise FileNotFoundError((block, weight_matches))
            raw = np.load(weight_matches[0]).astype("<f2", copy=False)
        else:
            raw = np.asarray(values).astype("<f2", copy=False)
        routing = json.loads((root / "block1-ffn-hidden-routing.json").read_text())
        # The SASS bias loads cover bytes [0x2c60,0x4c60), proving the old
        # archive-linear boundary was shifted by eight half slots.  The symbolic
        # QK C-operand trace gives a complete physical-half -> (query,key) map.
        layout_shift = 0x400 if block == 0 else (0x70 if block == 70 else 0)
        physical_bias = np.frombuffer(
            raw.tobytes(), dtype="<f2", count=4096, offset=0x2C60 + layout_shift
        )
        self.bias = np.empty((64, 64), dtype=np.float32)
        bias_routes = routing["bias_storage_half_to_qk"]
        if len(bias_routes) != 4096:
            raise AssertionError(len(bias_routes))
        for physical_half_text, destinations in bias_routes.items():
            if len(destinations) != 1:
                raise AssertionError((physical_half_text, destinations))
            storage_half = int(physical_half_text)
            query_token, key_token = destinations[0]
            self.bias[query_token, key_token] = physical_bias[storage_half]
        # Scale follows the bias immediately at weight_base+0x4c60.
        self.scale = float(np.frombuffer(
            raw.tobytes(), dtype="<f4", count=1, offset=0x4C60 + layout_shift
        )[0])
        attn_skip_physical = np.frombuffer(
            raw.tobytes(), dtype="<f2", count=32, offset=0x5070 + layout_shift
        ).astype(np.float16)
        self.attn_skip = attn_skip_physical[
            np.argsort(self.ffn_layout.output_to_qkv_k)
        ]

        route = routing["attended_component_n_to_projection_k"]
        self.attended_to_projection_k = np.empty((2, 16), dtype=np.int32)
        for component in range(2):
            for channel in range(16):
                values = route[f"component{component}_n{channel}"]
                if len(values) != 1:
                    raise AssertionError((component, channel, values))
                self.attended_to_projection_k[component, channel] = values[0]
        if len(np.unique(self.attended_to_projection_k)) != 32:
            raise AssertionError("attention->projection K routing is not a permutation")
        self.output_to_qkv_k = self.ffn_layout.output_to_qkv_k

    @staticmethod
    def _project(source: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return (source.astype(np.float32) @ matrix.T).astype(np.float16)

    def tile(
        self, canonical_e4m3: np.ndarray, attention_mask: np.ndarray | None = None
    ) -> np.ndarray:
        source_bytes = np.asarray(canonical_e4m3, dtype=np.uint8)
        if source_bytes.shape != (64, 32):
            raise ValueError(source_bytes.shape)
        qmma_bytes = self.ffn_layout.canonical_to_qmma(source_bytes)
        source = decode_e4m3(qmma_bytes).astype(np.float16)

        # SASS provenance groups the six chunks as Qe,Qo,Ke,Ko,Ve,Vo.  Q and K
        # remain 32-dimensional physical vectors (two 16-wide components).
        qe, qo = self._project(source, self.qe), self._project(source, self.qo)
        ke, ko = self._project(source, self.ke), self._project(source, self.ko)
        ve, vo = self._project(source, self.ve), self._project(source, self.vo)
        q = np.concatenate((qe, qo), axis=1).astype(np.float32)
        k = np.concatenate((ke, ko), axis=1).astype(np.float32)
        q_norm = np.sqrt(np.maximum(np.sum(q * q, axis=1, keepdims=True), 6.198883056640625e-05))
        k_norm = np.sqrt(np.maximum(np.sum(k * k, axis=1, keepdims=True), 6.198883056640625e-05))
        # SASS converts the FP32 scale to half, multiplies normalized Q, then
        # F2FP-quantizes Q/K before their QMMA.  Applying scale after matmul is
        # not equivalent across this quantization boundary.
        q_half = (q / q_norm * np.float16(self.scale)).astype(np.float16)
        k_half = (k / k_norm).astype(np.float16)
        q_quantized = quantize_half_e4m3(q_half).astype(np.float32)
        k_quantized = quantize_half_e4m3(k_half).astype(np.float32)

        logits = q_quantized @ k_quantized.T + self.bias
        if attention_mask is not None:
            mask = np.asarray(attention_mask, dtype=bool)
            if mask.shape != (64, 64):
                raise ValueError(mask.shape)
            logits = np.where(mask, -100.0, logits)
        logits -= np.max(logits, axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=1, keepdims=True)
        probability_quantized = quantize_half_e4m3(probability.astype(np.float16)).astype(np.float32)
        ve_quantized = quantize_half_e4m3(ve).astype(np.float32)
        vo_quantized = quantize_half_e4m3(vo).astype(np.float32)
        attended = [
            (probability_quantized @ ve_quantized).astype(np.float16),
            (probability_quantized @ vo_quantized).astype(np.float16),
        ]
        projection_input = np.empty((64, 32), dtype=np.float32)
        for component in range(2):
            projection_input[:, self.attended_to_projection_k[component]] = (
                quantize_half_e4m3(attended[component]).astype(np.float32)
            )

        residual_qkv = (source * self.attn_skip).astype(np.float16)
        residual = residual_qkv[:, self.output_to_qkv_k]
        with np.errstate(all="ignore"):
            output_n = (
                projection_input @ self.projection.T + residual.astype(np.float32)
            ).astype(np.float16)
        output_qkv = np.empty_like(output_n)
        output_qkv[:, self.output_to_qkv_k] = output_n
        output_bytes = encode_half_e4m3(output_qkv)
        return self.ffn_layout.qmma_to_canonical(output_bytes)


class Physical1HBlock:
    def __init__(
        self,
        root: Path = ROOT,
        block: int = 1,
        *,
        values: np.ndarray | None = None,
    ) -> None:
        self.block = block
        # Runtime phase offsets belong to the TinLayout address transform. Once
        # tensors are represented canonically they must not be rolled a second
        # time (block2/3 identity-oracle comparisons select no roll).
        self.shifted = False
        self.ffn = Physical1HFFN(root, block, values=values)
        self.attention = Physical1HAttention(root, block, values=values)

    def tile(
        self, canonical_e4m3: np.ndarray, attention_mask: np.ndarray | None = None
    ) -> np.ndarray:
        return self.attention.tile(self.ffn.tile(canonical_e4m3), attention_mask)

    @staticmethod
    def shifted_window_masks(height: int, width: int) -> np.ndarray:
        if height % 8 or width % 8:
            raise ValueError((height, width))
        region_map = np.zeros((height, width), dtype=np.int32)
        h_slices = (slice(0, -8), slice(-8, -4), slice(-4, None))
        w_slices = (slice(0, -8), slice(-8, -4), slice(-4, None))
        region = 0
        for hs in h_slices:
            for ws in w_slices:
                region_map[hs, ws] = region
                region += 1
        windows = (
            region_map.reshape(height // 8, 8, width // 8, 8)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 64)
        )
        return windows[:, :, None] != windows[:, None, :]

    def image(self, canonical_e4m3: np.ndarray) -> np.ndarray:
        source = np.asarray(canonical_e4m3, dtype=np.uint8)
        if source.ndim != 3 or source.shape[2] != 32:
            raise ValueError(source.shape)
        height, width, _ = source.shape
        if height % 8 or width % 8:
            raise ValueError("physical 1H image path currently requires H/W divisible by 8")
        working = np.roll(source, shift=(-4, -4), axis=(0, 1)) if self.shifted else source
        masks = self.shifted_window_masks(height, width) if self.shifted else None
        output = np.empty_like(working)
        window = 0
        for top in range(0, height, 8):
            for left in range(0, width, 8):
                tile = working[top : top + 8, left : left + 8].reshape(64, 32)
                mask = masks[window] if masks is not None else None
                if mask is not None:
                    permutation = self.ffn.token_permutation
                    mask = mask[np.ix_(permutation, permutation)]
                output[top : top + 8, left : left + 8] = self.tile(tile, mask).reshape(8, 8, 32)
                window += 1
        return np.roll(output, shift=(4, 4), axis=(0, 1)) if self.shifted else output


class TorchPhysical1HBlock(nn.Module):
    """Torch-native blocks1–3 using the recovered physical algorithm."""

    confidence = "SASS + controlled SM89 oracle algorithm; explicit FP16/E4M3 boundaries and standard Torch matmul"
    numeric_confidence = confidence

    def __init__(
        self,
        root: Path = ROOT,
        block: int = 1,
        *,
        quantize_output: bool = True,
        values: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        raw_values = (
            values.detach().cpu().numpy().astype("<f2", copy=False)
            if values is not None else None
        )
        recovered = Physical1HBlock(root, block, values=raw_values)
        self.block = block
        # The four runtime variants change physical TinLayout addressing/grid
        # bounds, not the canonical tensor's coordinates. Applying the old
        # Swin-style roll here duplicated that transform. Controlled identity
        # permutations plus real block2/3 outputs both select canonical no-roll.
        self.shift = (0, 0)
        self.shifted = False
        self.quantize_output = quantize_output
        ffn, attention = recovered.ffn, recovered.attention

        def buffer(name: str, value: np.ndarray | list[float]) -> None:
            self.register_buffer(name, torch.from_numpy(np.asarray(value).copy()))

        buffer("w1", np.stack(ffn.w1).astype(np.float32))
        buffer("w2", np.stack(ffn.w2).astype(np.float32))
        buffer("ffn_skip", ffn.skip.astype(np.float32))
        buffer("hidden_inverse", np.argsort(ffn.hidden_to_w2_k).astype(np.int64))
        buffer("output_route", ffn.output_to_qkv_k.astype(np.int64))
        buffer("output_inverse", np.argsort(ffn.output_to_qkv_k).astype(np.int64))
        buffer("token_permutation", ffn.token_permutation.astype(np.int64))
        buffer("token_inverse", np.argsort(ffn.token_permutation).astype(np.int64))
        buffer("channel_permutation", ffn.channel_permutation.astype(np.int64))
        buffer("channel_inverse", np.argsort(ffn.channel_permutation).astype(np.int64))
        for name in ("qe", "qo", "ke", "ko", "ve", "vo", "projection"):
            buffer(name, getattr(attention, name).astype(np.float32))
        buffer("bias", attention.bias.astype(np.float32))
        buffer("scale", np.asarray(attention.scale, dtype=np.float32))
        buffer("attn_skip", attention.attn_skip.astype(np.float32))
        attended_route = attention.attended_to_projection_k.reshape(-1)
        buffer("attended_inverse", np.argsort(attended_route).astype(np.int64))
        # Two native P×V K32 steps cover keys (0..15,48..63), then
        # (16..31,32..47), not the two contiguous halves of canonical key64.
        key16 = np.array([0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15], np.int64)
        buffer("probability_k_order", np.concatenate([key16 + k for k in (0,48,16,32)]))

    @staticmethod
    def _quantize(value: torch.Tensor) -> torch.Tensor:
        # Preserve the F16 input boundary of F2FP.SATFINITE.E4M3.F16.
        value = value.to(torch.float16).to(torch.float32)
        sign = torch.where(value < 0, -1.0, 1.0)
        absolute = value.abs()
        subnormal = torch.round(absolute * 512.0) / 512.0
        safe = absolute.clamp_min(torch.finfo(value.dtype).tiny)
        exponent = torch.floor(torch.log2(safe)).clamp(-6.0, 8.0)
        mantissa = torch.round((safe / torch.exp2(exponent) - 1.0) * 8.0)
        carry = mantissa >= 8.0
        exponent = exponent + carry.to(exponent.dtype)
        mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
        normal = torch.exp2(exponent) * (1.0 + mantissa / 8.0)
        result = torch.where(absolute < 0.015625, subnormal, normal)
        result = sign * result.clamp_max(448.0)
        return torch.where(value == 0, torch.zeros_like(result), result)

    @staticmethod
    def _activation(value: torch.Tensor) -> torch.Tensor:
        return fast_activation_2h(value)

    @staticmethod
    def _windows(value: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = value.shape
        return (
            value.reshape(batch, height // 8, 8, width // 8, 8, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 64, channels)
        )

    @staticmethod
    def _reverse_windows(
        value: torch.Tensor, batch: int, height: int, width: int
    ) -> torch.Tensor:
        channels = value.shape[-1]
        return (
            value.reshape(batch, height // 8, width // 8, 8, 8, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, channels)
        )

    def _mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        region_map = torch.zeros((height, width), dtype=torch.int64, device=device)
        shift_y, shift_x = self.shift
        h_slices = (
            (slice(0, -8), slice(-8, -shift_y), slice(-shift_y, None))
            if shift_y else (slice(None),)
        )
        w_slices = (
            (slice(0, -8), slice(-8, -shift_x), slice(-shift_x, None))
            if shift_x else (slice(None),)
        )
        region = 0
        for hs in h_slices:
            for ws in w_slices:
                region_map[hs, ws] = region
                region += 1
        windows = self._windows(region_map[None, :, :, None]).reshape(-1, 64)
        mask = windows[:, :, None] != windows[:, None, :]
        permutation = self.token_permutation
        return mask[:, permutation][:, :, permutation]

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[-1] != 32:
            raise ValueError(tuple(feature.shape))
        batch, original_height, original_width, _ = feature.shape
        height, width = (original_height + 7) & -8, (original_width + 7) & -8
        if (height, width) != (original_height, original_width):
            feature = F.pad(
                feature.permute(0, 3, 1, 2),
                (0, width - original_width, 0, height - original_height),
            ).permute(0, 2, 3, 1)
        # The pre/mixed-upsample/post blocks retain their FP16 adapter/fused input for the
        # residual; only W1 crosses the first E4M3 boundary. Ordinary blocks
        # already arrive as E4M3. SM89 pre:1e90/21d0; post:1a30/2c50.
        prequant_q = feature[..., self.channel_permutation] if self.block in (0,66,70) else None
        feature = self._quantize(feature)

        source_q = feature[..., self.channel_permutation]
        branches: list[torch.Tensor] = []
        for component in range(2):
            hidden = self._activation(matmul_fp16_linear(source_q, self.w1[component].T))
            hidden = self._quantize(hidden).to(feature.dtype)
            branches.append(hidden[..., self.hidden_inverse])
        # The skip is loaded in QKV-K order, while the W2 accumulator is still
        # indexed by output-N. Route QKV-K back to the matching output-N before
        # accumulation (the previous direct add silently permuted 16 channels).
        residual_q = (
            (prequant_q if prequant_q is not None else source_q).to(torch.float16)
            * self.ffn_skip.to(torch.float16)
        ).to(feature.dtype)
        residual = residual_q[..., self.output_route]
        # Native order is stream0/K0 -> stream1/K0 -> stream0/K1 ->
        # stream1/K1, with an FP16 accumulator between QMMA instructions.
        first_contribution = branches[0][..., :32] @ self.w2[0][:, :32].T
        # Ordinary 1H also uses the residual as the FIRST QMMA C operand:
        # runtime SM89 0xe10/0x1280/0x1720/0x1bb0 read the half-scaled input.
        # Moving it after all four instructions changes every FP16 boundary.
        output_n = (first_contribution + residual).to(torch.float16).to(feature.dtype)
        for component, begin in ((1, 0), (0, 32), (1, 32)):
            contribution = branches[component][..., begin : begin + 32] @ self.w2[component][:, begin : begin + 32].T
            output_n = (output_n + contribution).to(torch.float16).to(feature.dtype)
        ffn_half_q = output_n[..., self.output_inverse]
        ffn_q = self._quantize(ffn_half_q)
        ffn_canonical = ffn_q[..., self.channel_inverse]
        residual_canonical = ffn_half_q[..., self.channel_inverse]
        residual_working = (
            torch.roll(residual_canonical, shifts=(-self.shift[0], -self.shift[1]), dims=(1, 2))
            if self.shifted else residual_canonical
        )

        working = (
            torch.roll(ffn_canonical, shifts=(-self.shift[0], -self.shift[1]), dims=(1, 2))
            if self.shifted else ffn_canonical
        )
        # Runtime 512B cells are 2x8 stripes, arranged TL/BL/BR/TR.
        # Encoder phase origins join neighboring cells, not HWC roll windows.
        # Six-round controlled fingerprints establish members and native slots.
        phase_members = phase_valid = None
        if self.block in (2,3,4,67,68,69,70):
            phase_members, phase_valid, _ = phase_group_indices(
                height, width, ENCODER_1H_PHASES[self.block], device=feature.device
            )
            windows = working.reshape(batch, height*width, 32)[:, phase_members]
            windows = windows.masked_fill(~phase_valid[None,:,:,None], 0).reshape(-1,64,32)
            residual_windows = residual_working.reshape(batch, height*width, 32)[:, phase_members]
            residual_windows = residual_windows.masked_fill(~phase_valid[None,:,:,None], 0).reshape(-1,64,32)
        else:
            windows = self._windows(working)
            residual_windows = self._windows(residual_working)
        q_source = windows[:, self.token_permutation][:, :, self.channel_permutation]
        # QKV reads E4M3, but the attention residual retains the FFN's FP16
        # accumulator. All16 native projection C paths contain no F2FP boundary.
        residual_source = residual_windows[:, self.token_permutation][:, :, self.channel_permutation]
        qe, qo = (matmul_fp16_linear(q_source, self.qe.T)).to(torch.float16), (matmul_fp16_linear(q_source, self.qo.T)).to(torch.float16)
        ke, ko = (matmul_fp16_linear(q_source, self.ke.T)).to(torch.float16), (matmul_fp16_linear(q_source, self.ko.T)).to(torch.float16)
        ve, vo = (matmul_fp16_linear(q_source, self.ve.T)).to(torch.float16), (matmul_fp16_linear(q_source, self.vo.T)).to(torch.float16)
        query = torch.cat((qe, qo), dim=-1)
        key = torch.cat((ke, ko), dim=-1)
        # SM89 0x51f0..0x5340 converts the half norm to FP32, executes
        # MUFU.RSQ, then packs the reciprocal back to half before HMUL2.
        # Half sqrt followed by division introduces a different boundary.
        query = normalize_fp16_2h(query)
        key = normalize_fp16_2h(key)
        query = self._quantize((query * self.scale.half()).to(torch.float16)).to(feature.dtype)
        key = self._quantize(key.to(torch.float16)).to(feature.dtype)
        logits = query @ key.transpose(-1, -2) + self.bias
        if self.shifted:
            mask = self._mask(height, width, feature.device)
            mask = mask.repeat(batch, 1, 1)
            logits = logits.masked_fill(mask, -100.0)
        affine = affine_exp_input_2h(logits)
        exponent = fast_exp_2h(affine)
        probability = self._quantize(
            normalize_sum_fp16(exponent, denominator=sum_fp16_key64_1h(exponent))
        ).to(feature.dtype)
        value_e = self._quantize(ve.to(torch.float16)).to(feature.dtype)
        value_o = self._quantize(vo.to(torch.float16)).to(feature.dtype)
        probability_k = probability.index_select(-1, self.probability_k_order)
        value_e = value_e.index_select(-2, self.probability_k_order)
        value_o = value_o.index_select(-2, self.probability_k_order)
        attended_e = self._quantize((matmul_fp16_accumulate(probability_k, value_e)).to(torch.float16)).to(feature.dtype)
        attended_o = self._quantize((matmul_fp16_accumulate(probability_k, value_o)).to(torch.float16)).to(feature.dtype)
        projection_input = torch.cat((attended_e, attended_o), dim=-1)[..., self.attended_inverse]
        attention_residual_q = (
            residual_source.to(torch.float16) * self.attn_skip.to(torch.float16)
        ).to(feature.dtype)
        attention_residual = attention_residual_q[..., self.output_route]
        projected_n = (
            projection_input @ self.projection.T + attention_residual
        ).to(torch.float16).to(feature.dtype)
        output_q = projected_n[..., self.output_inverse]
        if self.quantize_output:
            output_q = self._quantize(output_q)
        output_windows = output_q[:, self.token_inverse][:, :, self.channel_inverse]
        if phase_members is not None:
            # Valid slots form a bijection: scatter once, without a Python CTA loop.
            output = torch.empty((batch,height*width,32), device=feature.device, dtype=feature.dtype)
            output[:,phase_members[phase_valid]] = output_windows.reshape(batch,-1,64,32)[:,phase_valid]
            output = output.reshape(batch,height,width,32)
        else:
            output = self._reverse_windows(output_windows, batch, height, width)
        if self.shifted:
            output = torch.roll(output, shifts=self.shift, dims=(1, 2))
        return output[:, :original_height, :original_width]


"""Torch-native physical 2H/64 fused-Swin family for blocks5–8/63–65.

The runtime record contains two 64->128->32->64 FFN streams. Attention contains
two 32-D Q/K/V heads selected by TID.Y/r162 and two K64->N32 projection halves.
The implementation decodes those physical E4M3 fragments directly, reproduces
the P32/P16 channel routes, the physical relative-bias layout, the half-bit LEA
exponential approximation, and the observed E4M3/FP16 boundaries. No fitted or
archive-linear proxy parameters are used.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent


def quantize_e4m3_2h(value: torch.Tensor) -> torch.Tensor:
    # Active SM89 F2FP.E4M3 converters consume F16, never F32.
    value = value.to(torch.float16).to(torch.float32)
    sign = torch.signbit(value)
    absolute = value.abs().clamp(max=448.0)
    tiny = torch.tensor(torch.finfo(torch.float32).tiny, device=value.device)
    exponent = torch.floor(torch.log2(absolute.clamp_min(tiny)))
    normal_exponent = exponent.clamp(-6.0, 8.0)
    step = torch.pow(2.0, normal_exponent - 3.0)
    normal = torch.round(absolute / step) * step
    subnormal = torch.round(absolute / (2.0 ** -9)) * (2.0 ** -9)
    rounded = torch.where(absolute < 2.0 ** -6, subnormal, normal).clamp(max=448.0)
    rounded = torch.where(sign, -rounded, rounded)
    return torch.where(absolute == 0, torch.zeros_like(rounded), rounded)


def matmul_fp16_accumulate(left: torch.Tensor, right: torch.Tensor, chunk: int = 32, *, initial: torch.Tensor | None = None) -> torch.Tensor:
    """QMMA.F16 output is read back as half by the next K32 instruction.

    PyTorch's ordinary half GEMM can retain a wider accumulator across K.
    Compute each physical K32 partial in FP32 and expose the documented
    FP16 boundary between instructions; no internal tensor-core bit emulation.
    """
    if left.shape[-1] != right.shape[-2]:
        raise ValueError((left.shape,right.shape))
    dtype=torch.promote_types(left.dtype,right.dtype)
    result = None if initial is None else initial.half().float()
    for k in range(0,left.shape[-1],chunk):
        partial=left[...,k:k+chunk].float() @ right[...,k:k+chunk,:].float()
        result=(partial if result is None else result+partial).half().float()
    if result is None:raise ValueError('empty MMA reduction')
    return result.to(dtype)


def matmul_fp16_linear(left: torch.Tensor, right: torch.Tensor, *, initial: torch.Tensor | None = None) -> torch.Tensor:
    # Some fused kernels start the projection accumulator with a half-scaled
    # residual. Adding that residual after the reduction is not equivalent.
    if left.shape[-1] <= 32 and initial is None:
        return left @ right
    return matmul_fp16_accumulate(left, right, initial=initial)


def matmul_fp16_splitk(left: torch.Tensor, right: torch.Tensor, partitions: int, *, initial: torch.Tensor | None = None) -> torch.Tensor:
    """Independent native split-K partials, then half-precision reduction.

    Only partition zero owns the residual. The fixed model uses this for ViT
    QKV (2x512), FFN contract (4x1024), and output projection (4x256).
    GPU atomic arrival order is not emulated; reduce partitions in index order.
    """
    if partitions < 1 or left.shape[-1] % partitions:
        raise ValueError((left.shape, partitions))
    size = left.shape[-1] // partitions
    total = None
    for part in range(partitions):
        k = part * size
        partial = matmul_fp16_accumulate(left[...,k:k+size], right[...,k:k+size,:],
                                        initial=initial if part == 0 else None)
        total = partial.half() if total is None else total + partial.half()
    return total.to(torch.promote_types(left.dtype,right.dtype))


def normalize_fp16_2h(value: torch.Tensor) -> torch.Tensor:
    half = value.to(torch.float16)
    if half.shape[-1] != 32:
        raise ValueError(f'physical Q/K normalization expects head dimension32, got {half.shape[-1]}')
    # Per lane: HMUL2 on dims16/24, then HFMA2 on dims0/8. The
    # butterfly adds xor2, xor1, then the two packed half lanes.
    # Keep the second square fused with the first rounded square.
    lo, hi = half[..., :8].float(), half[..., 8:16].float()
    a = (lo * lo + (half[..., 16:24] * half[..., 16:24]).float()).half()
    b = (hi * hi + (half[..., 24:32] * half[..., 24:32]).float()).half()
    c = a + b
    u = (c[..., :2] + c[..., 4:6]) + (c[..., 2:4] + c[..., 6:8])
    norm2 = (u[..., :1] + u[..., 1:2]).clamp_min(6.198883056640625e-05)
    inv = torch.rsqrt(norm2.to(torch.float32)).to(torch.float16)
    return (half * inv).to(torch.float16)


_SUM64_KEY_ORDERS = {
    'vit': (0, 8, 16, 24, 32, 40, 48, 56, 2, 10, 18, 26, 34, 42, 50, 58, 4, 12, 20, 28, 36, 44, 52, 60, 6, 14, 22, 30, 38, 46, 54, 62, 1, 9, 17, 25, 33, 41, 49, 57, 3, 11, 19, 27, 35, 43, 51, 59, 5, 13, 21, 29, 37, 45, 53, 61, 7, 15, 23, 31, 39, 47, 55, 63),
    '16h': (0, 1, 16, 17, 32, 33, 48, 49, 4, 5, 20, 21, 36, 37, 52, 53, 8, 9, 24, 25, 40, 41, 56, 57, 12, 13, 28, 29, 44, 45, 60, 61, 2, 3, 18, 19, 34, 35, 50, 51, 6, 7, 22, 23, 38, 39, 54, 55, 10, 11, 26, 27, 42, 43, 58, 59, 14, 15, 30, 31, 46, 47, 62, 63),
    '2h': (0, 8, 16, 24, 32, 40, 48, 56, 2, 10, 18, 26, 34, 42, 50, 58, 4, 12, 20, 28, 36, 44, 52, 60, 6, 14, 22, 30, 38, 46, 54, 62, 1, 9, 17, 25, 33, 41, 49, 57, 3, 11, 19, 27, 35, 43, 51, 59, 5, 13, 21, 29, 37, 45, 53, 61, 7, 15, 23, 31, 39, 47, 55, 63),
    '4h': (0, 16, 8, 24, 32, 48, 40, 56, 4, 20, 12, 28, 36, 52, 44, 60, 1, 17, 9, 25, 33, 49, 41, 57, 5, 21, 13, 29, 37, 53, 45, 61, 2, 18, 10, 26, 34, 50, 42, 58, 6, 22, 14, 30, 38, 54, 46, 62, 3, 19, 11, 27, 35, 51, 43, 59, 7, 23, 15, 31, 39, 55, 47, 63),
    '8h': (0, 16, 4, 20, 32, 48, 36, 52, 2, 18, 6, 22, 34, 50, 38, 54, 8, 24, 12, 28, 40, 56, 44, 60, 10, 26, 14, 30, 42, 58, 46, 62, 1, 17, 5, 21, 33, 49, 37, 53, 3, 19, 7, 23, 35, 51, 39, 55, 9, 25, 13, 29, 41, 57, 45, 61, 11, 27, 15, 31, 43, 59, 47, 63),
}


def sum_fp16_key64(value: torch.Tensor, family: str) -> torch.Tensor:
    """Exact family-specific pair/four/four/half FP16 probability reduction.

    Key orders are from independent bias-coordinate/SASS expression traces,
    not fitted numerical choices. Each family covers all64 keys exactly once.
    """
    h = value.half()
    if h.shape[-1] != 64:
        raise ValueError(f'{family} probability reduction expects64 keys, got {h.shape[-1]}')
    ordered = h[..., list(_SUM64_KEY_ORDERS[family])].reshape(*h.shape[:-1], 2, 4, 4, 2)
    pairs = ordered[...,0] + ordered[...,1]
    groups = ((pairs[...,0] + pairs[...,1]) + pairs[...,2]) + pairs[...,3]
    halves = ((groups[...,0] + groups[...,1]) + groups[...,2]) + groups[...,3]
    return halves[...,0:1] + halves[...,1:2]


def sum_fp16_key64_1h(value: torch.Tensor) -> torch.Tensor:
    """1H key64 reduction, proved by exact QK/SEL/SHFL expression tracing."""
    h = value.half()
    if h.shape[-1] != 64:
        raise ValueError(f'1H probability reduction expects64 keys, got {h.shape[-1]}')
    g = (h[...,0:8] + h[...,8:16]) + (h[...,48:56] + h[...,56:64])
    g = g + (h[...,16:24] + h[...,24:32])
    g = g + (h[...,32:40] + h[...,40:48])
    packed = ((g[...,0:2] + g[...,2:4]) + g[...,4:6]) + g[...,6:8]
    return packed[...,0:1] + packed[...,1:2]


def normalize_sum_fp16(value: torch.Tensor, *, denominator: torch.Tensor | None = None) -> torch.Tensor:
    """Half sum -> FP32 reciprocal -> half reciprocal -> half product.

    This is the shared MUFU.RCP/F2FP/HMUL2 probability-normalization path;
    direct half division does not expose the reciprocal's half boundary.
    """
    half = value.half()
    total = (half.sum(dim=-1, keepdim=True) if denominator is None else denominator.half()).clamp_min(6.198883056640625e-05)
    inverse = torch.reciprocal(total.float()).half()
    return half * inverse


def fast_exp_2h(value: torch.Tensor) -> torch.Tensor:
    """SM89 half2 LEA exponential approximation used by 2H attention."""
    half = value.to(torch.float16).contiguous()
    bits = half.view(torch.int16).to(torch.int32) & 0xFFFF
    encoded = ((bits << 5) + 0x8000) & 0xFFFF
    return encoded.to(torch.int16).view(torch.float16)


def affine_exp_input_2h(logits: torch.Tensor) -> torch.Tensor:
    """The shared Swin HFMA2 affine, with exactly one FP16 rounding."""
    return (logits.half().float() * 0.044921875 + 1.30078125).half().clamp(1.03125,1.5693359375)


def fast_activation_2h(value: torch.Tensor) -> torch.Tensor:
    """Fused-Swin gate: clamp only the polynomial input, not its final source."""
    original = value.to(torch.float16)
    clipped = original.clamp(-4.0, 4.0).float()
    # HFMA2 rounds once after multiply+add. Separate half multiplications and
    # additions inserted extra rounding boundaries throughout every family.
    gate = (clipped.abs() * -0.055908203125 + 0.447265625).half().float()
    gate = (clipped * gate + 0.89453125).half()
    # SM89 0x1260..0x12e0 preserves the original QMMA accumulator in R40
    # and uses the clamped copy R62 only to form the gate.
    return original * gate


def window_partition_2h(value: torch.Tensor) -> torch.Tensor:
    batch, height, width, heads, channels = value.shape
    return (
        value.reshape(batch, height // 8, 8, width // 8, 8, heads, channels)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(-1, heads, 64, channels)
    )


def window_reverse_2h(value: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    heads, channels = value.shape[1], value.shape[-1]
    return (
        value.reshape(batch, height // 8, width // 8, heads, 8, 8, channels)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .reshape(batch, height, width, heads * channels)
    )


def shift_mask_2h(
    height: int, width: int, device: torch.device, shift: tuple[int, int]
) -> torch.Tensor:
    region = torch.zeros((height, width), dtype=torch.int64, device=device)
    shift_y, shift_x = shift
    hs = ((slice(0, -8), slice(-8, -shift_y), slice(-shift_y, None)) if shift_y else (slice(None),))
    ws = ((slice(0, -8), slice(-8, -shift_x), slice(-shift_x, None)) if shift_x else (slice(None),))
    index = 0
    for h in hs:
        for w in ws:
            region[h, w] = index
            index += 1
    windows = (
        region.reshape(height // 8, 8, width // 8, 8)
        .permute(0, 2, 1, 3)
        .reshape(-1, 64)
    )
    return windows[:, :, None].ne(windows[:, None, :])


def _decode_subtile_2h(raw: bytes, offset: int) -> tuple[np.ndarray, np.ndarray]:
    tile = np.frombuffer(raw[offset : offset + 512], dtype=np.uint8).reshape(32, 16)
    return unpack_b_bytes(tile[:, :8]), unpack_b_bytes(tile[:, 8:])


def _n_concat_2h(raw: bytes, offsets: list[int]) -> np.ndarray:
    return np.concatenate(
        [fragment for offset in offsets for fragment in _decode_subtile_2h(raw, offset)],
        axis=1,
    )


def _k_concat_2h(raw: bytes, groups: list[list[int]]) -> np.ndarray:
    return np.concatenate([_n_concat_2h(raw, offsets) for offsets in groups], axis=0)


class TorchPhysical2HBlock(nn.Module):
    """Real archive FFN plus statically recovered physical 2H attention."""

    confidence = (
        "SM89 physical dual-stream 64->128->32->64 FFN + "
        "SASS-direct physical 2H attention"
    )
    numeric_confidence = confidence

    def __init__(
        self,
        root: Path = ROOT,
        block: int = 5,
        *,
        values: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if block not in (5, 6, 7, 8, 62, 63, 64, 65):
            raise ValueError(block)
        if values is None:
            matches = sorted((root / "weights").glob(f"*_block{block}_layer0_layer.npy"))
            if len(matches) != 1:
                raise FileNotFoundError(matches)
            raw_half = np.load(matches[0]).astype("<f2", copy=False)
        else:
            raw_half = values.detach().cpu().numpy().astype("<f2", copy=False)
        if raw_half.nbytes not in (61760, 69936):
            raise ValueError(raw_half.nbytes)
        raw = raw_half.tobytes()
        # Exact SM89 loop decomposition. r114 selects two independent streams:
        # 64->128 W1, 128->32 main, then 32->64 tail. The old direct FP16
        # [96,64]/[64,96] view was only a container-shaped proxy.
        w1, w2_main, w2_tail = [], [], []
        for stream in range(2):
            b1 = 0x2000 * stream
            bm = 0x4000 + 0x1000 * stream
            bt = 0x6000 + 0x0800 * stream
            w1.append(decode_e4m3(_k_concat_2h(raw, [
                list(range(b1 + 0x0000, b1 + 0x1000, 0x200)),
                list(range(b1 + 0x1000, b1 + 0x2000, 0x200)),
            ]).T))
            w2_main.append(decode_e4m3(_k_concat_2h(raw, [
                [bm + 0x000, bm + 0x200],
                [bm + 0x400, bm + 0x600],
                [bm + 0x800, bm + 0xA00],
                [bm + 0xC00, bm + 0xE00],
            ]).T))
            w2_tail.append(decode_e4m3(_n_concat_2h(raw, [
                bt + 0x000, bt + 0x200, bt + 0x400, bt + 0x600,
            ]).T))
        self.register_buffer("w1", torch.from_numpy(np.stack(w1).astype(np.float32)))
        self.register_buffer("w2_main", torch.from_numpy(np.stack(w2_main).astype(np.float32)))
        self.register_buffer("w2_tail", torch.from_numpy(np.stack(w2_tail).astype(np.float32)))
        p32 = np.asarray([
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
            16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31,
        ], dtype=np.int64)
        q32 = np.argsort(p32)
        self.register_buffer("w1_to_main", torch.from_numpy(np.concatenate([q32 + 32 * i for i in range(4)])))
        self.register_buffer("main_to_tail", torch.from_numpy(q32))
        output_route = np.concatenate([p32, p32 + 32])
        self.register_buffer("ffn_output_route", torch.from_numpy(output_route))
        # Decoder blocks63-65 repeat the encoder block6-8 physical phase
        # family.  Use the recovered M/N and token routes of the corresponding
        # phase; block ids themselves do not define the layout.
        layout_block = {63: 6, 64: 7, 65: 8}.get(block, block)
        # Canonical tile element -> fused-kernel (M,K/N) coordinates.  The
        # shifted XY phase interleaves token and channel bits; treating each
        # canonical pixel as an independent C64 vector is therefore invalid.
        tile_token = np.repeat(np.arange(64, dtype=np.int64), 64)
        tile_channel = np.tile(np.arange(64, dtype=np.int64), 64)
        if layout_block in (6, 7, 8):
            output_n_index = (
                ((tile_channel >> 0) & 1) << 0
                | ((tile_token >> 0) & 1) << 1
                | ((tile_token >> 1) & 1) << 2
                | ((tile_channel >> 1) & 1) << 3
                | ((tile_channel >> 3) & 1) << 4
                | ((tile_token >> 5) & 1) << 5
            )
            matrix_m_index = (
                ((tile_token >> 2) & 1) << 0
                | ((tile_token >> 4) & 1) << 1
                | ((tile_channel >> 4) & 1) << 2
                | ((tile_channel >> 5) & 1) << 3
                | ((tile_channel >> 2) & 1) << 4
                | ((tile_token >> 3) & 1) << 5
            )
            if layout_block in (7, 8):
                matrix_m_index = (
                    ((tile_token >> 2) & 1) << 0
                    | ((tile_token >> 4) & 1) << 1
                    | ((tile_channel >> 2) & 1) << 2
                    | ((tile_channel >> 4) & 1) << 3
                    | ((tile_token >> 3) & 1) << 4
                    | ((tile_channel >> 5) & 1) << 5
                )
        else:
            output_n_index = np.argsort(output_route)[tile_channel]
            matrix_m_index = tile_token
        matrix_k_index = output_route[output_n_index]
        if np.unique(matrix_m_index * 64 + matrix_k_index).size != 4096:
            raise AssertionError(f"block{block}: non-bijective 2H input layout")
        self.register_buffer("matrix_m_index", torch.from_numpy(matrix_m_index))
        self.register_buffer("matrix_k_index", torch.from_numpy(matrix_k_index))
        self.register_buffer("output_n_index", torch.from_numpy(output_n_index))
        self.register_buffer("ffn_skip", torch.from_numpy(np.frombuffer(raw, dtype="<f2", count=64, offset=0x7010).astype(np.float32)))
        bias_storage = np.frombuffer(
            raw, dtype="<f2", count=8192, offset=0xA0A0
        ).astype(np.float32).reshape(2, 4096)
        query = np.arange(64, dtype=np.int64)[:, None]
        key = np.arange(64, dtype=np.int64)[None, :]
        bias_physical = (
            (query // 16) * 1024
            + (query % 8) * 32
            + ((query % 16) // 8) * 2
            + (key // 16) * 256
            + ((key % 16) // 8) * 4
            + ((key % 8) // 2) * 8
            + (key % 2)
        )
        self.register_buffer(
            "bias", torch.from_numpy(bias_storage[:, bias_physical].copy())
        )
        token = np.arange(64, dtype=np.int64)
        query_token_route = (
            ((token >> 0) & 1) << 1
            | ((token >> 1) & 1) << 4
            | ((token >> 2) & 1) << 0
            | ((token >> 3) & 1) << 2
            | ((token >> 4) & 1) << 5
            | ((token >> 5) & 1) << 3
        )
        key_token_route = (
            ((token >> 0) & 1) << 0
            | ((token >> 1) & 1) << 4
            | ((token >> 2) & 1) << 1
            | ((token >> 3) & 1) << 2
            | ((token >> 4) & 1) << 5
            | ((token >> 5) & 1) << 3
        )
        if block == 5:
            # inpview's all64 one-hot bias queries and independent V-key
            # signatures share one real-pixel order, unlike the generic kernel.
            query_token_route = key_token_route.copy()
        if layout_block == 6:
            phase_token_route = (
                ((token >> 0) & 1) << 0
                | ((token >> 1) & 1) << 2
                | ((token >> 2) & 1) << 1
                | ((token >> 3) & 1) << 4
                | ((token >> 4) & 1) << 3
                | ((token >> 5) & 1) << 5
            )
            query_token_route = phase_token_route
            key_token_route = phase_token_route
        elif layout_block == 7:
            phase_token_route = (
                ((token >> 0) & 1) << 0
                | ((token >> 1) & 1) << 4
                | ((token >> 2) & 1) << 1
                | ((token >> 3) & 1) << 2
                | ((token >> 4) & 1) << 3
                | ((token >> 5) & 1) << 5
            )
            query_token_route = phase_token_route
            key_token_route = phase_token_route
        elif layout_block == 8:
            phase_token_route = (
                ((token >> 0) & 1) << 0
                | ((token >> 1) & 1) << 4
                | ((token >> 2) & 1) << 1
                | ((token >> 3) & 1) << 2
                | ((token >> 4) & 1) << 5
                | ((token >> 5) & 1) << 3
            )
            query_token_route = phase_token_route
            key_token_route = phase_token_route
        if block in (6, 7, 8):
            # These are native CTA/QKV rows after the global gather below,
            # not the old arbitrary per-canonical-tile matrix rows.
            # Independent V-key basis and all64 query labels give identity.
            query_token_route = token.copy()
            key_token_route = token.copy()
        self.register_buffer("attention_query_token_route", torch.from_numpy(query_token_route))
        self.register_buffer("attention_key_token_route", torch.from_numpy(key_token_route))
        self.register_buffer("attn_skip", torch.from_numpy(np.frombuffer(raw, dtype="<f2", count=64, offset=0xF0B0).astype(np.float32)))
        attn_skip_route = np.asarray([
            0, 1, 2, 9, 3, 4, 7, 11, 8, 5, 10, 13, 6, 12, 14, 15,
            16, 17, 18, 22, 21, 19, 23, 24, 20, 26, 27, 29, 25, 28, 30, 31,
            32, 33, 34, 35, 37, 38, 42, 43, 36, 40, 39, 41, 45, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 55, 62, 63,
        ], dtype=np.int64)
        self.register_buffer("attn_skip_for_channel", torch.from_numpy(
            np.frombuffer(raw, dtype="<f2", count=64, offset=0xF0B0).astype(np.float32)[attn_skip_route]
        ))

        # r162 selects two complete 32-D attention heads. Each head has six
        # N16 Q/K/V chunks at a 0xc00 stride; the old implementation mistook
        # head0's low/high N16 chunks for two separate 16-D heads and omitted
        # head1 entirely.
        def qkv16(low: int) -> np.ndarray:
            return decode_e4m3(_k_concat_2h(raw, [[low], [low + 0x1800]]).T)

        q_heads, k_heads, v_heads, projection_heads = [], [], [], []
        for head in range(2):
            qbase = 0x0C00 * head
            q_heads.append(np.concatenate([
                qkv16(0x70A0 + qbase), qkv16(0x72A0 + qbase)
            ], axis=0))
            k_heads.append(np.concatenate([
                qkv16(0x74A0 + qbase), qkv16(0x76A0 + qbase)
            ], axis=0))
            v_heads.append(np.concatenate([
                qkv16(0x78A0 + qbase), qkv16(0x7AA0 + qbase)
            ], axis=0))
            pbase = 0xE0B0 + 0x0400 * head
            projection_heads.append(decode_e4m3(_k_concat_2h(raw, [
                [pbase + 0x000, pbase + 0x200],
                [pbase + 0x800, pbase + 0xA00],
            ]).T))
        self.register_buffer("q_weight", torch.from_numpy(np.stack(q_heads).astype(np.float32)))
        self.register_buffer("k_weight", torch.from_numpy(np.stack(k_heads).astype(np.float32)))
        self.register_buffer("v_weight", torch.from_numpy(np.stack(v_heads).astype(np.float32)))
        self.register_buffer("projection_weight", torch.from_numpy(
            np.stack(projection_heads).astype(np.float32)
        ))
        self.register_buffer("scale", torch.from_numpy(
            np.frombuffer(raw, dtype="<f4", count=2, offset=0xE0A0).astype(np.float32).copy()
        ))
        self.register_buffer(
            "projection_input_route",
            torch.from_numpy(np.concatenate([q32, q32 + 32])),
        )
        self.register_buffer("projection_output_route", torch.from_numpy(output_route.copy()))
        self.block = block
        # inpview's matrix rows follow 2-wide vertical stripes in each
        # real8x8 window, not row-major pixels. Native one-hot key/address
        # probes give M bits -> real-pixel bits (0,3,4,5,1,2).
        m = np.arange(64, dtype=np.int64)
        inpview_rows = (m & 1) | ((m & 14) << 2) | ((m >> 3) & 6)
        self.register_buffer("inpview_m_to_pixel", torch.from_numpy(inpview_rows))
        self.register_buffer("inpview_pixel_to_m", torch.from_numpy(np.argsort(inpview_rows)))
        phase_start = 5 if block <= 8 else 62
        self.shift = ((0, 0), (4, 4), (0, 4), (4, 0))[(block - phase_start) % 4]
        if layout_block in (6, 7, 8):
            # This is a phase-specific FFN row/channel view, not an image
            # window. Attention applies the real shift in _attention_cta_rows,
            # not by rolling this canonical tensor as if it were spatial HWC.
            self.shift = (0, 0)
        self.shifted = any(self.shift)

    def _to_internal_tiles(self, feature: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = feature.shape
        canonical = (
            feature.reshape(batch, height // 8, 8, width // 8, 8, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 4096)
        )
        internal = torch.empty(
            (canonical.shape[0], 64, 64), dtype=feature.dtype, device=feature.device
        )
        internal[:, self.matrix_m_index, self.matrix_k_index] = canonical
        return internal

    def _from_internal_tiles(
        self, internal_n: torch.Tensor, batch: int, height: int, width: int
    ) -> torch.Tensor:
        canonical = internal_n[:, self.matrix_m_index, self.output_n_index]
        return (
            canonical.reshape(batch, height // 8, width // 8, 8, 8, 64)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, 64)
        )

    def _attention_cta_rows(self, height: int, width: int, device: torch.device):
        """Generic2H shifted CTA rows in the current FFN matrix-row view.

        SM89 input addresses + FFN first-C/shared dataflow independently map
        native QKV M bits to packet-byte bits (6,7,8,2,10,11). Its K order
        already matches _to_internal_tiles; only whole rows cross tile groups.
        Quarter-plane byte layout here is the exact tinlayout_2h64 input codec.
        """
        sx, sy = {6: (-4, -4), 7: (-4, 0), 8: (0, -4)}[self.block]
        gx, gy = (width-sx+7)//8, (height-sy+7)//8
        group = torch.arange(gx*gy, device=device)[:, None]
        m = torch.arange(64, device=device)[None, :]
        qx = (8*(group % gx)+sx)//4 + ((m >> 4) & 1)
        qy = (8*(group // gx)+sy)//4 + (m >> 5)
        valid = (qx >= 0) & (qx < width//4) & (qy >= 0) & (qy < height//4)
        raw = (qy*(width//4)+qx)*1024 + (m & 7)*64 + ((m >> 3) & 1)*4
        raw = raw.masked_fill(~valid, 0)
        cell, within = raw//512, raw%512
        plane = cell//((height//8)*(width//4))
        cy = (cell % ((height//8)*(width//4)))//(width//4)
        cx = cell % (width//4)
        token = (cx%2)*32 + within//16
        channel = plane*16 + within%16
        row = (cy*(width//8)+cx//2)*64 + self.matrix_m_index[token*64+channel]
        return row, valid

    def forward(self, feature: torch.Tensor, *, quantize_output: bool = True) -> torch.Tensor:
        batch, original_height, original_width, channels = feature.shape
        if channels != 64:
            raise ValueError((self.block, channels))
        height, width = (original_height + 7) & -8, (original_width + 7) & -8
        if (height, width) != (original_height, original_width):
            feature = F.pad(
                feature.permute(0, 3, 1, 2),
                (0, width - original_width, 0, height - original_height),
            ).permute(0, 2, 3, 1)

        feature = quantize_e4m3_2h(feature)
        internal_feature = self._to_internal_tiles(feature)
        contributions = []
        for stream in range(2):
            hidden = quantize_e4m3_2h(
                fast_activation_2h(matmul_fp16_linear(internal_feature, self.w1[stream].T))
            ).to(feature.dtype)
            main_input = hidden.index_select(-1, self.w1_to_main)
            bottleneck = quantize_e4m3_2h(
                matmul_fp16_linear(main_input, self.w2_main[stream].T)
            ).to(feature.dtype)
            tail_input = bottleneck.index_select(-1, self.main_to_tail)
            contributions.append(matmul_fp16_linear(tail_input, self.w2_tail[stream].T))
        # Tail accumulators are output-N ordered. Shifted phase variants map
        # canonical token/channel bits jointly into M and N.
        residual_n = (
            internal_feature.index_select(-1, self.ffn_output_route).half() * self.ffn_skip.half()
        ).float()
        # SM89 0x0b50.. initializes the tail accumulators from the skip;
        # r114 then visits stream0 and stream1, each with one K32 tail.
        output_n = residual_n
        for contribution in contributions:
            output_n = (output_n + contribution).half().float()
        output_n = quantize_e4m3_2h(output_n)
        feature = self._from_internal_tiles(output_n, batch, height, width)
        attention_feature = feature
        if self.shifted:
            attention_feature = torch.roll(
                attention_feature, shifts=(-self.shift[0], -self.shift[1]), dims=(1, 2)
            )
        if self.block == 5:
            # The first2H dispatch is inpview_fp8, not the ordinary kernel.
            # Its compact input view groups consecutive64 pooled pixels into
            # our provisional8x8 tiles. Attention instead visits real spatial
            # 8x8 windows (inpview CTA X/Y *8). Restore that view before QKV.
            attention_feature = (attention_feature.reshape(batch,height//8,8,width//8,8,64)
                                 .permute(0,1,3,2,4,5).reshape(batch,height,width,64))
        physical = self._to_internal_tiles(attention_feature)
        cta_rows = cta_valid = None
        if self.block in (6, 7, 8):
            cta_rows, cta_valid = self._attention_cta_rows(height, width, feature.device)
            physical = physical.reshape(batch, height*width, 64)[:, cta_rows]
            physical = physical.masked_fill(~cta_valid[None,:,:,None], 0).reshape(-1,64,64)
        if self.block == 5:
            physical = physical.index_select(-2, self.inpview_m_to_pixel)

        q = torch.stack([
            (matmul_fp16_linear(physical, self.q_weight[head].T)).to(torch.float16) for head in range(2)
        ], dim=-2)
        k = torch.stack([
            (matmul_fp16_linear(physical, self.k_weight[head].T)).to(torch.float16) for head in range(2)
        ], dim=-2)
        v = torch.stack([
            (matmul_fp16_linear(physical, self.v_weight[head].T)).to(torch.float16) for head in range(2)
        ], dim=-2)
        q = quantize_e4m3_2h(
            normalize_fp16_2h(q)
            * self.scale.to(torch.float16).reshape(1, 1, 2, 1)
        )
        k = quantize_e4m3_2h(normalize_fp16_2h(k))
        v = quantize_e4m3_2h(v)

        qw = q.permute(0, 2, 1, 3).index_select(-2, self.attention_query_token_route)
        kw = k.permute(0, 2, 1, 3).index_select(-2, self.attention_key_token_route)
        vw = v.permute(0, 2, 1, 3).index_select(-2, self.attention_key_token_route)
        logits = matmul_fp16_accumulate(qw, kw.transpose(-1, -2), initial=self.bias.half().unsqueeze(0))
        # The phase mask is applied in fused M coordinates. Interior windows are
        # unaffected; boundary routing is retained as a separate phase concern.
        affine = affine_exp_input_2h(logits)
        weight = fast_exp_2h(affine)
        probability = quantize_e4m3_2h(
            normalize_sum_fp16(weight, denominator=sum_fp16_key64(weight, "2h"))
        )
        attended_sass = quantize_e4m3_2h(
            matmul_fp16_accumulate(probability.to(torch.float16), vw.to(torch.float16))
        )
        attended_heads = torch.empty_like(attended_sass)
        attended_heads[..., self.attention_query_token_route, :] = attended_sass
        attended = attended_heads.permute(0, 2, 1, 3).reshape(-1, 64, 64)

        projection_input = attended.index_select(-1, self.projection_input_route)
        attention_residual_n = (
            physical.index_select(-1, self.ffn_output_route).half() * self.attn_skip.half()
        )
        projected_n = torch.cat([
            matmul_fp16_linear(projection_input, self.projection_weight[head].T,
                               initial=attention_residual_n[...,head*32:(head+1)*32])
            for head in range(2)
        ], dim=-1)
        attention_output_n = quantize_e4m3_2h(projected_n) if quantize_output else projected_n.half().float()
        if self.block == 5:
            attention_output_n = attention_output_n.index_select(-2, self.inpview_pixel_to_m)
        if cta_rows is not None:
            # All valid native rows form one global bijection. Virtual rows
            # participate as zero Q/K/V but are not published to the tensor.
            published = torch.empty((batch,height*width,64), device=feature.device, dtype=attention_output_n.dtype)
            published[:,cta_rows[cta_valid]] = attention_output_n.reshape(batch,-1,64,64)[:,cta_valid]
            attention_output_n = published.reshape(-1,64,64)
        output = self._from_internal_tiles(attention_output_n, batch, height, width)
        if self.shifted:
            output = torch.roll(output, shifts=self.shift, dims=(1, 2))
        if self.block == 5:
            # Return the compact view expected by the recovered first2H
            # publication; this is the inverse of the attention view above.
            output = (output.reshape(batch,height//8,width//8,8,8,64)
                      .permute(0,1,3,2,4,5).reshape(batch,height,width,64))
        return output[:, :original_height, :original_width]


    def forward_with_prequant(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # DS publishes E4M3 skip but pools the original half accumulator.
        prequant = self.forward(feature, quantize_output=False)
        return quantize_e4m3_2h(prequant), prequant


class TorchPhysicalDecoder2HBlock(TorchPhysical2HBlock):
    """Decoder 2H body in its recovered fused token/channel representation.

    Block62 skip-identity signatures map the decoder raw tensor back to the
    encoder block8 HWC boundary.  Blocks63-65 then recover the full canonical
    element -> (physical token, output-N) bijection, while controlled uniform
    attention and learned-bias probes recover groups plus independent Q/K
    slots.  This avoids the old ordinary HWC roll approximation.
    """

    confidence = "controlled-runtime decoder2H fused layout and attention groups"
    numeric_confidence = confidence

    def __init__(self, root: Path, block: int, values: torch.Tensor | None = None) -> None:
        if block not in (62, 63, 64, 65):
            raise ValueError(block)
        super().__init__(root, block, values=values)
        p32 = np.asarray([
            0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
            16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
        ], dtype=np.int64)
        p64 = np.concatenate([p32, p32 + 32])
        self.register_buffer("decoder_n_to_k", torch.from_numpy(np.argsort(p64).copy()))
        self.decoder_phase = {62: (0, 0), 63: (4, 4), 64: (0, 4), 65: (4, 0)}[block]

    @staticmethod
    def _indices(height:int,width:int) -> tuple[np.ndarray,np.ndarray]:
        if height%16 or width%16:raise ValueError((height,width))
        r,c,n=np.indices((height,width,64),dtype=np.int64)
        output_n=(n&1)|((c&1)<<1)|(((c>>1)&1)<<2)|(((n>>1)&1)<<3)|(((n>>3)&1)<<4)|(((r>>2)&1)<<5)
        low=((n>>2)&1)|(((c>>2)&1)<<1)|((r&1)<<2)|(((r>>1)&1)<<3)|(((c>>3)&1)<<4)
        token=low+32*(c//16+(width//16)*(r//8+(height//8)*(n//16)))
        return token.ravel(),output_n.ravel()

    @staticmethod
    def _deposit_slot(slot: np.ndarray, mapping: tuple[tuple[int, int], ...]) -> np.ndarray:
        return sum(((slot >> source) & 1) << target for source, target in mapping)

    def _window_layout(self, height: int, width: int, device: torch.device):
        token = np.arange(height * width, dtype=np.int64)
        low=token&63;outer=token>>6;columns=width//16
        row=(outer//columns)*4+((low>>3)&1)+2*(low&1)
        col=(outer%columns)*16+((low>>1)&1)+2*((low>>2)&1)+4*((low>>4)&1)+8*((low>>5)&1)
        sy, sx = self.decoder_phase
        keys = np.stack(((row + sy) // 8, (col + sx) // 8), axis=1)
        _, group = np.unique(keys, axis=0, return_inverse=True)
        standard_slot = ((row + sy) & 7) * 8 + ((col + sx) & 7)
        query_slot = self._deposit_slot(
            standard_slot, ((0,0),(1,1),(3,2),(4,3),(2,4),(5,5))
        )
        # Correcting the V/projection channel permutation in the controlled
        # bias basis shows decoder XY uses the same physical token route for
        # Q and K (the initially decoded distinct key route was basis-order).
        key_slot = query_slot.copy()
        groups = int(group.max()) + 1
        query_members = np.zeros((groups, 64), dtype=np.int64)
        key_members = np.zeros((groups, 64), dtype=np.int64)
        query_valid = np.zeros((groups, 64), dtype=np.bool_)
        key_valid = np.zeros((groups, 64), dtype=np.bool_)
        for current in range(token.size):
            g = group[current]
            qs, ks = query_slot[current], key_slot[current]
            if query_valid[g, qs] or key_valid[g, ks]:
                raise AssertionError("duplicate decoder2H attention slot")
            query_members[g, qs] = current
            key_members[g, ks] = current
            query_valid[g, qs] = True
            key_valid[g, ks] = True
        return tuple(torch.from_numpy(value).to(device=device) for value in (
            query_members, key_members, query_valid, key_valid
        ))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, original_height, original_width, channels = feature.shape
        if channels != 64:
            raise ValueError((self.block, channels))
        height,width=original_height,original_width
        if height%16 or width%16:raise ValueError((height,width))
        token_np, output_n_np = self._indices(height, width)
        token = torch.from_numpy(token_np).to(device=feature.device)
        output_n_index = torch.from_numpy(output_n_np).to(device=feature.device)
        source = quantize_e4m3_2h(feature).reshape(batch, -1)
        internal_n = torch.empty(
            (batch, height * width, 64), dtype=source.dtype, device=source.device
        )
        internal_n[:, token, output_n_index] = source
        input_k = internal_n.index_select(-1, self.decoder_n_to_k)

        contributions = []
        for stream in range(2):
            hidden = quantize_e4m3_2h(
                fast_activation_2h(matmul_fp16_linear(input_k, self.w1[stream].T))
            ).to(feature.dtype)
            main_input = hidden.index_select(-1, self.w1_to_main)
            bottleneck = quantize_e4m3_2h(
                matmul_fp16_linear(main_input, self.w2_main[stream].T)
            ).to(feature.dtype)
            tail_input = bottleneck.index_select(-1, self.main_to_tail)
            contributions.append(matmul_fp16_linear(tail_input, self.w2_tail[stream].T))
        tail = (internal_n.half() * self.ffn_skip.half()).float()
        for contribution in contributions:
            tail = (tail + contribution).half().float()
        internal_n = quantize_e4m3_2h(tail)
        input_k = internal_n.index_select(-1, self.decoder_n_to_k)
        q = torch.stack([
            (matmul_fp16_linear(input_k, self.q_weight[head].T)).to(torch.float16) for head in range(2)
        ], dim=2)
        k = torch.stack([
            (matmul_fp16_linear(input_k, self.k_weight[head].T)).to(torch.float16) for head in range(2)
        ], dim=2)
        v = torch.stack([
            (matmul_fp16_linear(input_k, self.v_weight[head].T)).to(torch.float16) for head in range(2)
        ], dim=2)
        q = quantize_e4m3_2h(
            normalize_fp16_2h(q) * self.scale.to(torch.float16).reshape(1,1,2,1)
        )
        k = quantize_e4m3_2h(normalize_fp16_2h(k))
        v = quantize_e4m3_2h(v)
        qm, km, qvalid, kvalid = self._window_layout(height, width, feature.device)
        qw = q[:, qm].permute(0,1,3,2,4).masked_fill(
            ~qvalid[None,:,None,:,None], 0.0
        )
        kw = k[:, km].permute(0,1,3,2,4).masked_fill(
            ~kvalid[None,:,None,:,None], 0.0
        )
        vw = v[:, km].permute(0,1,3,2,4).masked_fill(
            ~kvalid[None,:,None,:,None], 0.0
        )
        logits = matmul_fp16_accumulate(qw, kw.transpose(-1,-2), initial=self.bias.half().unsqueeze(0).unsqueeze(0))
        affine = affine_exp_input_2h(logits)
        weight = fast_exp_2h(affine)
        probability = quantize_e4m3_2h(
            normalize_sum_fp16(weight, denominator=sum_fp16_key64(weight, "2h"))
        )
        attended_windows = quantize_e4m3_2h(
            matmul_fp16_accumulate(probability.to(torch.float16), vw.to(torch.float16))
        ).permute(0,1,3,2,4)
        attended_heads = torch.empty_like(q)
        active = qvalid.reshape(-1)
        attended_heads[:, qm.reshape(-1)[active]] = attended_windows.reshape(
            batch,-1,2,32
        )[:,active]
        attended = attended_heads.reshape(batch,height*width,64)
        projection_input = attended.index_select(-1,self.projection_input_route)
        residual_n = internal_n.half() * self.attn_skip.half()
        projected_n = torch.cat([
            matmul_fp16_linear(projection_input, self.projection_weight[head].T,
                               initial=residual_n[...,head*32:(head+1)*32])
            for head in range(2)
        ], dim=-1)
        internal_n = quantize_e4m3_2h(projected_n)
        output = internal_n[:,token,output_n_index].reshape(batch,height,width,64)
        return output[:,:original_height,:original_width]


__all__ = ["TorchPhysical2HBlock", "TorchPhysicalDecoder2HBlock"]


"""SM89-backed physical Torch implementation of the 4H/128 fused family."""

from pathlib import Path


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
_P32 = np.asarray([
    0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
    16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
], dtype=np.int64)


def _bit_route(bits: tuple[int, ...]) -> np.ndarray:
    token = np.arange(64, dtype=np.int64)
    return sum(((token >> source) & 1) << target for source, target in enumerate(bits))


class TorchPhysical4HBlock(nn.Module):
    """Physical blocks9-14 path recovered from the runtime SM89 CUBIN.

    Each record contains four 128->128->32 FFN branches followed by one
    128->128 contraction. The attention uses phase-specific fused-coordinate
    windows and the recovered 12-bit physical relative-bias swizzle.
    """

    confidence = "controlled-runtime blocks9-13 physical FFN/attention"
    numeric_confidence = confidence

    def __init__(self, block: int, values: torch.Tensor | None = None) -> None:
        super().__init__()
        if block not in (*range(9, 15), *range(56, 62)):
            raise ValueError("TorchPhysical4HBlock supports encoder blocks9-14 and decoder blocks56-61")
        if values is None:
            matches = sorted((ROOT / "weights").glob(f"*_block{block}_layer0_layer.npy"))
            if len(matches) != 1:
                raise FileNotFoundError(matches)
            raw_half = np.load(matches[0]).astype("<f2", copy=False)
        else:
            raw_half = values.detach().cpu().numpy().astype("<f2", copy=False)
        raw = raw_half.tobytes()
        if block == 14 and len(raw) >= 197184:
            raw = raw[:197184]
        if len(raw) != 197184:
            raise ValueError(f"block{block}: expected at least 197184 body bytes, got {len(raw)}")

        w1, w2a = [], []
        for stream in range(4):
            w1.append(decode_e4m3(_k_concat_2h(raw, [
                [
                    stream * 0x4000 + k_plane * 0x1000 + hidden * 0x400 + half * 0x200
                    for hidden in range(4) for half in range(2)
                ]
                for k_plane in range(4)
            ])))
            w2a.append(decode_e4m3(_k_concat_2h(raw, [
                [0x10000 + stream * 0x1000 + k_plane * 0x400 + half * 0x200
                 for half in range(2)]
                for k_plane in range(4)
            ])))
        w2b = decode_e4m3(_k_concat_2h(raw, [
            [0x14000 + k_plane * 0x1000 + output * 0x400 + half * 0x200
             for output in range(4) for half in range(2)]
            for k_plane in range(4)
        ]))
        self.register_buffer("w1", torch.from_numpy(np.stack(w1).astype(np.float32)))
        self.register_buffer("w2a", torch.from_numpy(np.stack(w2a).astype(np.float32)))
        self.register_buffer("w2b", torch.from_numpy(w2b.astype(np.float32)))
        self.register_buffer("ffn_skip", torch.from_numpy(
            np.frombuffer(raw, "<f2", count=128, offset=0x18010).astype(np.float32).copy()
        ))

        q, k, v = [], [], []
        for head in range(4):
            matrix = decode_e4m3(_k_concat_2h(raw, [
                [0x18120 + k_plane * 0x3000 + head * 0x0C00 + tile * 0x200
                 for tile in range(6)]
                for k_plane in range(4)
            ]))
            q.append(matrix[:, :32]); k.append(matrix[:, 32:64]); v.append(matrix[:, 64:96])
        self.register_buffer("q_weight", torch.from_numpy(np.stack(q).astype(np.float32)))
        self.register_buffer("k_weight", torch.from_numpy(np.stack(k).astype(np.float32)))
        self.register_buffer("v_weight", torch.from_numpy(np.stack(v).astype(np.float32)))
        bias_storage = np.frombuffer(
            raw, "<f2", count=4 * 4096, offset=0x24120
        ).astype(np.float32).reshape(4, 4096)
        query = np.arange(64, dtype=np.int64)[:, None]
        key = np.arange(64, dtype=np.int64)[None, :]
        # Twelve controlled binary-address probes recover the exact 4H bias
        # tensor swizzle (all 4096 logical Q/K entries form a full bijection).
        query_bits = (7, 5, 6, 10, 1, 11)
        key_bits = (4, 0, 3, 8, 2, 9)
        bias_physical = sum(
            ((query >> source) & 1) << target
            for source, target in enumerate(query_bits)
        ) + sum(
            ((key >> source) & 1) << target
            for source, target in enumerate(key_bits)
        )
        self.register_buffer(
            "bias", torch.from_numpy(bias_storage[:, bias_physical].copy())
        )
        self.register_buffer("scale", torch.from_numpy(
            np.frombuffer(raw, "<f4", count=4, offset=0x2C120).copy()
        ))
        projection = decode_e4m3(_k_concat_2h(raw, [
            [0x2C130 + k_plane * 0x1000 + output * 0x400 + half * 0x200
             for output in range(4) for half in range(2)]
            for k_plane in range(4)
        ]))
        self.register_buffer("projection", torch.from_numpy(projection.copy()))
        self.register_buffer("attn_skip", torch.from_numpy(
            np.frombuffer(raw, "<f2", count=128, offset=0x30130).astype(np.float32).copy()
        ))
        p128 = np.concatenate([_P32 + 32 * group for group in range(4)])
        self.register_buffer("channel_route", torch.from_numpy(np.argsort(p128).copy()))
        token_routes = {
            9: ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
            10: ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
            11: ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
            12: ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
            13: ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
            14: ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
        }
        query_bits, key_bits = token_routes.get(block, token_routes[9])
        self.register_buffer("query_token_route", torch.from_numpy(_bit_route(query_bits)))
        self.register_buffer("key_token_route", torch.from_numpy(_bit_route(key_bits)))
        bias_routes: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        bias_query_bits, bias_key_bits = bias_routes.get(
            block, ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
        )
        self.register_buffer("bias_query_route", torch.from_numpy(_bit_route(bias_query_bits)))
        self.register_buffer("bias_key_route", torch.from_numpy(_bit_route(bias_key_bits)))
        self.block = block
        # Decoder 4H runtime phases 57..61 are Y, none, XY, X, Y in the
        # encoder physical convention.  Reuse the independently recovered
        # phase formulas rather than the old DECODER_SHIFTED offset.
        self.layout_block = {
            57: 12,
            58: 13,
            59: 14,
            60: 11,
            61: 12,
        }.get(block, block)

    def _window_layout(
        self, tile_height: int, tile_width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block=self.layout_block;sx=block in (10,11,14);sy=block in (10,12,14)
        if tile_height%2 or tile_width%2:raise ValueError((tile_height,tile_width))
        nh=tile_height//2+int(sy);nw=tile_width//2+int(sx)
        members=np.zeros((nh*nw,64),np.int64);valid=np.zeros_like(members,dtype=bool)
        for ty in range(tile_height):
            for tx in range(tile_width):
                xhalf=tx//2+(tile_width//2)*(ty&1);side_x=xhalf&1;ax=tile_width//2-1-xhalf//2
                for local in range(16):
                    # Form the linear quarter-plane address before splitting
                    # its bits. An odd number of half-bands carries into the
                    # slot bit as well as the window coordinate.
                    y=ty//2+(tile_height//2)*(((local>>1)&1)+2*((local>>3)&1))
                    yhalf=y//2;side_y=yhalf&1;ay=yhalf//2
                    gy=ay+(side_y if sy else 0);gx=ax+(1-side_x if sx else 0)
                    right=1-side_x if sx else side_x;top=1-side_y if sy else side_y
                    slot=(local&1)|(((local>>2)&1)<<1)|((tx&1)<<2)|(right<<3)|((y&1)<<4)|(top<<5)
                    g=gy*nw+gx
                    if valid[g,slot]:raise AssertionError('duplicate4H slot')
                    members[g,slot]=(ty*tile_width+tx)*16+local;valid[g,slot]=True
        return torch.from_numpy(members).to(device),torch.from_numpy(valid).to(device)

    def forward(self, feature: torch.Tensor, *, quantize_output: bool = True) -> torch.Tensor:
        batch, original_height, original_width, channels = feature.shape
        if channels != 128:
            raise ValueError(f"block9: expected C=128, got {channels}")
        height,width=original_height,original_width
        if height%8 or width%8:raise ValueError((height,width))

        source = quantize_e4m3_2h(feature)
        source_k = source.index_select(-1, self.channel_route)
        parts = []
        for stream in range(4):
            hidden_n = quantize_e4m3_2h(fast_activation_2h(matmul_fp16_linear(source_k, self.w1[stream])))
            hidden_k = hidden_n.index_select(-1, self.channel_route)
            parts.append(matmul_fp16_linear(hidden_k, self.w2a[stream]))
        middle_k = quantize_e4m3_2h(
            torch.cat(parts, dim=-1).index_select(-1, self.channel_route)
        )
        feature = quantize_e4m3_2h(
            matmul_fp16_linear(middle_k, self.w2b, initial=source.half() * self.ffn_skip.half())
        )

        tile_height, tile_width = height // 4, width // 4
        rows = (
            feature.reshape(batch, tile_height, 4, tile_width, 4, 128)
            .permute(0,1,3,2,4,5).reshape(batch, tile_height * tile_width * 16, 128)
        )
        members, valid = self._window_layout(tile_height, tile_width, feature.device)
        windows = rows[:, members]
        windows = windows.masked_fill(~valid[None, :, :, None], 0.0)
        windows = windows.reshape(-1, 64, 128)
        qkv_input = windows.index_select(-1, self.channel_route)
        q = torch.stack([(matmul_fp16_linear(qkv_input, self.q_weight[h])).to(torch.float16) for h in range(4)], dim=1)
        k = torch.stack([(matmul_fp16_linear(qkv_input, self.k_weight[h])).to(torch.float16) for h in range(4)], dim=1)
        v = torch.stack([(matmul_fp16_linear(qkv_input, self.v_weight[h])).to(torch.float16) for h in range(4)], dim=1)
        q = quantize_e4m3_2h(normalize_fp16_2h(q) * self.scale.to(torch.float16).reshape(1,4,1,1))
        k = quantize_e4m3_2h(normalize_fp16_2h(k)); v = quantize_e4m3_2h(v)
        qw = q.index_select(-2, self.query_token_route)
        kw = k.index_select(-2, self.key_token_route)
        vw = v.index_select(-2, self.key_token_route)
        routed_bias = self.bias.index_select(-2, self.bias_query_route).index_select(
            -1, self.bias_key_route
        )
        logits = matmul_fp16_accumulate(qw, kw.transpose(-1,-2), initial=routed_bias.half().unsqueeze(0))
        # Boundary slots are zero-padded before QKV, but the kernel still applies
        # all 64 learned bias entries and keeps them in the softmax denominator.
        affine = affine_exp_input_2h(logits)
        weight = fast_exp_2h(affine)
        probability = quantize_e4m3_2h(normalize_sum_fp16(weight, denominator=sum_fp16_key64(weight, "4h")))
        attended_sass = quantize_e4m3_2h(matmul_fp16_accumulate(probability.to(torch.float16), vw.to(torch.float16)))
        attended_heads = torch.empty_like(attended_sass)
        attended_heads[..., self.query_token_route, :] = attended_sass
        attended = attended_heads.permute(0,2,1,3).reshape(-1,64,128).index_select(-1,self.channel_route)
        projected_n = matmul_fp16_linear(attended, self.projection, initial=windows.half() * self.attn_skip.half())
        values = quantize_e4m3_2h(projected_n) if quantize_output else projected_n.half().float()
        output_windows = values.reshape(batch, -1, 64, 128)
        output_rows = torch.empty_like(rows)
        for window in range(members.shape[0]):
            active = valid[window]
            output_rows[:, members[window, active]] = output_windows[:, window, active]
        output = (
            output_rows.reshape(batch,tile_height,tile_width,4,4,128)
            .permute(0,1,3,2,4,5).reshape(batch,height,width,128)
        )
        return output[:, :original_height, :original_width]


    def forward_with_prequant(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prequant = self.forward(feature, quantize_output=False)
        return quantize_e4m3_2h(prequant), prequant


__all__ = ["TorchPhysical4HBlock"]


"""Runtime-SM89 recovered physical 8H/256 fused-block primitives."""
from pathlib import Path

import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _unpack_qmma_b_e4m3(tile: bytes) -> np.ndarray:
    source=np.frombuffer(tile,dtype=np.uint8).reshape(32,16)
    return np.stack((unpack_b_bytes(source[:,:8]),unpack_b_bytes(source[:,8:])))


def _decode_e4m3_numpy(value: np.ndarray) -> np.ndarray:
    result=decode_e4m3(np.asarray(value,dtype=np.uint8)).astype(np.float32)
    if np.isnan(result).any(): raise ValueError("unexpected E4M3 NaN")
    return result


def _decode_packed_qmma_projection(raw: bytes,start: int,input_width: int,output_width: int) -> np.ndarray:
    planes=[];plane_bytes=32*output_width
    for plane in range(input_width//32):
        base=start+plane*plane_bytes
        fragments=[_unpack_qmma_b_e4m3(raw[o:o+512]) for o in range(base,base+plane_bytes,512)]
        planes.append(np.concatenate([f[h] for f in fragments for h in range(2)],axis=1))
    return _decode_e4m3_numpy(np.concatenate(planes,axis=0))


def _fragment(raw: bytes, offset: int, half: int) -> np.ndarray:
    packed = _unpack_qmma_b_e4m3(raw[offset:offset + 0x200])
    return _decode_e4m3_numpy(packed[half])


def decode_8h_qkv(raw: bytes) -> np.ndarray:
    """Decode eight interleaved 256x(Q32,K32,V32) SM89 head matrices."""
    qkv = np.empty((8, 3, 256, 32), np.float32)
    for head in range(8):
        for plane in range(8):
            for component in range(3):
                for q in range(4):
                    offset = (0x58220 + plane * 0x6000 + head * 0xC00
                              + component * 0x400 + (q >> 1) * 0x200)
                    qkv[head, component, 32*plane:32*plane+32, 8*q:8*q+8] = _fragment(raw, offset, q & 1)
    return qkv


def decode_8h_ffn(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode SM89's 8 x (256->128->32) grouped FFN plus 256 projection."""
    expansion = np.empty((8, 256, 128), np.float32)
    reduce = np.empty((8, 128, 32), np.float32)
    output = np.empty((256, 256), np.float32)
    for g in range(8):
        for j in range(8):
            for t in range(4):
                for q in range(4):
                    off = g * 0x8000 + j * 0x1000 + t * 0x400 + (q >> 1) * 0x200
                    expansion[g, 32*j:32*j+32, 32*t+8*q:32*t+8*q+8] = _fragment(raw, off, q & 1)
        for t in range(4):
            for q in range(4):
                off = 0x40000 + g * 0x1000 + t * 0x400 + (q >> 1) * 0x200
                reduce[g, 32*t:32*t+32, 8*q:8*q+8] = _fragment(raw, off, q & 1)
    for j in range(8):
        for g in range(8):
            for q in range(4):
                off = 0x48000 + j * 0x2000 + g * 0x400 + (q >> 1) * 0x200
                output[32*j:32*j+32, 32*g+8*q:32*g+8*q+8] = _fragment(raw, off, q & 1)
    return expansion, reduce, output


class TorchPhysical8HFFN(nn.Module):
    confidence = "runtime-SM89 8H FFN; full 8x(256->128->32)->256 topology and stage routes oracle-closed"
    _P32 = np.array([0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31], dtype=np.int64)

    @staticmethod
    def _route_bits(value: int, bits: int) -> int:
        destinations = (0,3,1,2,4,5,6,7)
        return sum(((value >> bit) & 1) << destinations[bit] for bit in range(bits))

    def __init__(self, root: str | Path, block: int = 15, *, values: torch.Tensor | None = None) -> None:
        super().__init__(); root = Path(root)
        if values is None:
            matches = glob.glob(str(root / "weights" / f"*_block{block}_layer0_layer.npy"))
            if len(matches) != 1: raise FileNotFoundError((block, matches))
            raw = np.load(matches[0]).astype("<f2", copy=False).tobytes()
        else:
            raw = values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        expansion, reduce, output = decode_8h_ffn(raw)
        q32 = np.argsort(self._P32)
        q256 = np.concatenate([q32 + 32 * group for group in range(8)])
        reduce_effective = np.empty_like(reduce)
        for group in range(8):
            for n in range(32):
                for k in range(128):
                    source = q32[n] ^ self._route_bits(k ^ n, 7)
                    reduce_effective[group, source, n] = reduce[group, k, n]
        output_effective = np.empty_like(output)
        for n in range(256):
            for k in range(256):
                source = q256[n] ^ self._route_bits(k ^ n, 8)
                output_effective[source, n] = output[k, n]
        self.register_buffer("expansion", torch.from_numpy(expansion))
        self.register_buffer("reduce", torch.from_numpy(reduce_effective))
        self.register_buffer("output", torch.from_numpy(output_effective))
        self.register_buffer("input_route", torch.from_numpy(q256))
        self.register_buffer("skip", torch.from_numpy(np.frombuffer(raw, "<f2", count=256, offset=0x58010).astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape; x = x.reshape(-1, 256)
        source = x.index_select(-1, self.input_route)
        groups = []
        for group in range(8):
            hidden = quantize_e4m3_2h(
                fast_activation_2h(matmul_fp16_linear(source, self.expansion[group]))
            )
            groups.append(quantize_e4m3_2h(matmul_fp16_linear(hidden, self.reduce[group])))
        reduced = torch.cat(groups, dim=-1)
        y = quantize_e4m3_2h(matmul_fp16_linear(reduced, self.output, initial=x.half() * self.skip.half()))
        return y.reshape(original_shape)


class TorchPhysical8HBlock(nn.Module):
    """Block15-family 8-head physical fused Swin recovered from SM89."""
    confidence = "runtime-SM89 physical 8H path; encoder and decoder bodies controlled-oracle closed"
    numeric_confidence = confidence

    def __init__(
        self,
        root: str | Path,
        block: int = 15,
        phase: tuple[int, int] | None = None,
        *, values: torch.Tensor | None = None,
    ) -> None:
        super().__init__(); root = Path(root); self.block = block
        self.phase = phase if phase is not None else ((0,0),(4,4),(0,4),(4,0))[(block-15)%4]
        self.ffn = TorchPhysical8HFFN(root, block, values=values)
        if values is None:
            matches = glob.glob(str(root / "weights" / f"*_block{block}_layer0_layer.npy"))
            if len(matches) != 1: raise FileNotFoundError((block, matches))
            raw = np.load(matches[0]).astype("<f2", copy=False).tobytes()
        else:
            raw = values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        self.register_buffer("qkv", torch.from_numpy(decode_8h_qkv(raw)))
        # The 8H bias is one 32768-half slab, not eight ordinary 64x64
        # row-major matrices. Controlled address-bit probes recover the full
        # (head, query slot, key slot) address permutation.
        bias_raw=np.frombuffer(raw,"<f2",count=8*4096,offset=0x88220).astype(np.float32)
        qbits=(5,6,10,7,1,11);kbits=(0,3,8,4,2,9)
        def deposit(value:int,bits:tuple[int,...])->int:
            return sum(((value>>i)&1)<<bit for i,bit in enumerate(bits))
        bias_index=np.array([[[ (head<<12)+deposit(q,qbits)+deposit(k,kbits) for k in range(64)] for q in range(64)] for head in range(8)])
        self.register_buffer("bias",torch.from_numpy(bias_raw[bias_index].copy()))
        self.register_buffer("scale", torch.from_numpy(np.frombuffer(raw,"<f4",count=8,offset=0x98220).copy()))
        projection = _decode_packed_qmma_projection(raw,0x98240,256,256)
        q256 = self.ffn.input_route.detach().cpu().numpy()
        effective = np.empty_like(projection)
        for n in range(256):
            for k in range(256):
                source = q256[n] ^ TorchPhysical8HFFN._route_bits(k ^ n, 8)
                effective[source,n] = projection[k,n]
        self.register_buffer("projection", torch.from_numpy(effective))
        self.register_buffer("attn_skip", torch.from_numpy(np.frombuffer(raw,"<f2",count=256,offset=0xA8240).astype(np.float32).copy()))

    @staticmethod
    def valid_token_indices(height:int,width:int) -> np.ndarray:
        """Rank-compressed SM89 view; local axes cross the outer tile axes.

        Gray512 (36x32) identity proves all294912 publication addresses and
        the partial final native row removes local_x>=4 at base_y=4.
        Appending four ordinary tensor rows is not this storage layout.
        """
        ph=(height+7)&-8;pw=(width+7)&-8;wh,ww=ph//8,pw//8
        row,col=np.indices((ph,pw));by,ly=row%wh,row//wh;bx,lx=col%ww,col//ww
        return np.flatnonzero(((by*8+lx<height)&(bx*8+ly<width)).ravel())

    @classmethod
    def compact_plane_indices(cls, height:int, width:int) -> np.ndarray:
        """4H producer quarter-plane token read by each valid 8H token.

        Order the native addresses, excluding absent half-window slots. This
        reduces to the rectangular column interleave for complete groups, but
        does not truncate the group count or insert pixels for partial groups.
        """
        ph=(height+7)&-8;pw=(width+7)&-8;ww=pw//8
        index=cls.valid_token_indices(height,width)
        row,col=index//pw,index%pw
        address=(col%ww)*(ph*8)+row*8+col//ww
        return np.argsort(np.argsort(address))

    @classmethod
    def expand_valid_tokens(cls,feature:torch.Tensor) -> torch.Tensor:
        b,h,w,c=feature.shape;ph=(h+7)&-8;pw=(w+7)&-8
        if (ph,pw)==(h,w):return feature
        ids=torch.from_numpy(cls.valid_token_indices(h,w)).to(feature.device)
        expanded=feature.new_zeros((b,ph*pw,c));expanded[:,ids]=feature.reshape(b,h*w,c)
        return expanded.reshape(b,ph,pw,c)

    def _members(self, height: int, width: int) -> tuple[np.ndarray,np.ndarray]:
        ph=(height+7)&-8;pw=(width+7)&-8;wh,ww=ph//8,pw//8
        grid=np.full(ph*pw,-1,np.int64);grid[self.valid_token_indices(height,width)]=np.arange(height*width);grid=grid.reshape(ph,pw)
        phase=self.phase
        groups: dict[tuple[int,int],list[tuple[int,int]]]={}
        for row in range(ph):
            base_y,local_y=row%wh,row//wh
            for col in range(pw):
                if grid[row,col]<0:continue
                base_x,local_x=col%ww,col//ww
                group_y=base_y+(1 if phase[1] and local_x>=4 else 0)
                group_x=base_x+(1 if phase[0] and local_y>=4 else 0)
                # Shifted kernels retain the full 64-slot window. Boundary
                # groups therefore contain holes at their physical slots;
                # packing valid tokens to the front changes learned-bias rows.
                slot_y=(local_y+phase[0])&7
                slot_x=(local_x+phase[1])&7
                groups.setdefault((group_y,group_x),[]).append((slot_y*8+slot_x,int(grid[row,col])))
        ordered=[groups[key] for key in sorted(groups)]
        members=np.zeros((len(ordered),64),np.int64);valid=np.zeros((len(ordered),64),bool)
        for i,group in enumerate(ordered):
            for slot,token in group:
                if valid[i,slot]:raise AssertionError("duplicate 8H window slot")
                members[i,slot]=token;valid[i,slot]=True
        return members,valid

    def attention(self, feature: torch.Tensor, *, quantize_output: bool = True) -> torch.Tensor:
        batch,height,width,channels=feature.shape
        if channels != 256: raise ValueError(channels)
        rows=feature.reshape(-1,256); source=rows.index_select(-1,self.ffn.input_route)
        qkv=torch.stack([matmul_fp16_linear(source, self.qkv[h, c]) for h in range(8) for c in range(3)])
        qkv=qkv.reshape(8,3,batch*height*width,32).permute(1,2,0,3);q,k,v=qkv
        q=quantize_e4m3_2h(normalize_fp16_2h(q)*self.scale.half().reshape(1,8,1))
        k=quantize_e4m3_2h(normalize_fp16_2h(k));v=quantize_e4m3_2h(v)
        member_np,valid_np=self._members(height,width);members=torch.from_numpy(member_np).to(feature.device);valid=torch.from_numpy(valid_np).to(feature.device)
        # Each batch owns the same physical window topology.
        members=(members.unsqueeze(0)+torch.arange(batch,device=feature.device).reshape(-1,1,1)*(height*width)).reshape(-1,64)
        valid=valid.unsqueeze(0).expand(batch,-1,-1).reshape(-1,64)
        qw=q[members].permute(0,2,1,3)*valid[:,None,:,None];kw=k[members].permute(0,2,1,3)*valid[:,None,:,None];vw=v[members].permute(0,2,1,3)*valid[:,None,:,None]
        logits=matmul_fp16_accumulate(qw,kw.transpose(-1,-2),initial=self.bias.unsqueeze(0).half())
        affine = affine_exp_input_2h(logits)
        exponent=fast_exp_2h(affine)
        probability=quantize_e4m3_2h(normalize_sum_fp16(exponent, denominator=sum_fp16_key64(exponent, "8h")))
        attended=quantize_e4m3_2h(matmul_fp16_accumulate(probability.to(torch.float16),vw.to(torch.float16)))
        restored=torch.empty((batch*height*width,8,32),device=feature.device,dtype=attended.dtype)
        values=attended.permute(0,2,1,3);restored[members[valid]]=values[valid]
        output=matmul_fp16_linear(restored.reshape(-1, 256), self.projection, initial=rows.half() * self.attn_skip.half())
        if quantize_output:
            output = quantize_e4m3_2h(output)
        return output.reshape(batch,height,width,256)

    def forward_with_prequant(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the published body and its pre-E4M3 final accumulator.

        Downsample block22 feeds the latter directly into its fused pooling path;
        publishing and decoding the body output first introduces an extra E4M3
        boundary that the CUBIN does not have.
        """
        feature=self.ffn(feature)
        prequant=self.attention(feature,quantize_output=False)
        published=quantize_e4m3_2h(prequant)
        return published,prequant

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        published,_=self.forward_with_prequant(feature)
        return published


__all__ = ["decode_8h_ffn","decode_8h_qkv","TorchPhysical8HFFN","TorchPhysical8HBlock"]


"""Packed-E4M3 physical candidate for DLSS5 blocks 23..30 split-Swin16H."""

import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_P32=np.array([0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31],np.int64)

def _deposit(value:int,positions:tuple[int,...])->int:
    return sum(((value>>bit)&1)<<position for bit,position in enumerate(positions))

def _decode_group_factor(raw:bytes,base:int,k_positions:tuple[int,...],n_positions:tuple[int,...],rep_positions:tuple[int,...])->np.ndarray:
    """Decode the four 64-wide factor paths used by split-Swin FFN-expand."""
    u=np.frombuffer(raw,np.uint8)
    out=np.empty((4,8,64,64),np.float32)
    for rep in range(4):
        rb=_deposit(rep,rep_positions)
        for group in range(8):
            gn=group*64
            for k in range(64):
                kb=_deposit(k,k_positions)
                offsets=[base+rb+kb+_deposit(gn+n,n_positions) for n in range(64)]
                out[rep,group,k]=decode_e4m3(u[offsets])
    return out

def _raw(root:Path,block:int,layer:int)->bytes:
 m=glob.glob(str(root/'weights'/f'*block{block}_layer{layer}_layer.npy'))
 if len(m)!=1:raise FileNotFoundError((block,layer,m))
 return np.load(m[0]).astype('<f2',copy=False).tobytes()

class TorchPhysicalSplitSwin16H(nn.Module):
    confidence='physical split-Swin path validated against NVIDIA boundaries for encoder blocks23-30 and decoder blocks40-47, including block30 pooled terminal projection'
    def __init__(self,root:str|Path,block:int):
        super().__init__();root=Path(root);self.block=block
        r0,r1,r2,r3=(_raw(root,block,i) for i in range(4))
        if tuple(map(len,(r0,r1,r2,r3)))!=(0x80000,0x40400,0xE0040,0x40400):raise ValueError((block,tuple(map(len,(r0,r1,r2,r3)))))
        self.factorized_expand=(23<=block<=30) or (40<=block<=47)
        if self.factorized_expand:
            self.register_buffer('expand_input',torch.from_numpy(_decode_packed_qmma_projection(r0,0,512,512)))
            self.register_buffer('expand_factor_in',torch.from_numpy(_decode_group_factor(r0,0x40000,(0,1,4,5,2,13),(6,3,7,8,9,10,14,15,16),(11,12))))
            self.register_buffer('expand_factor_out',torch.from_numpy(_decode_group_factor(r0,0x60000,(0,1,4,5,2,11),(6,7,8,3,9,10,14,15,16),(12,13))))
            self.gate=None;self.value=None
        else:
            self.register_buffer('gate',torch.from_numpy(_decode_packed_qmma_projection(r0,0,512,512)))
            self.register_buffer('value',torch.from_numpy(_decode_packed_qmma_projection(r0,0x40000,512,512)))
            self.expand_input=None;self.expand_factor_in=None;self.expand_factor_out=None
        self.register_buffer('ffn_projection',torch.from_numpy(_decode_packed_qmma_projection(r1,0,512,512)))
        self.register_buffer('ffn_skip',torch.from_numpy(np.frombuffer(r1,'<f2',count=512,offset=0x40000).astype(np.float32).copy()))
        self.register_buffer('qkv',torch.from_numpy(_decode_packed_qmma_projection(r2,0,512,1536)))
        bias_raw=np.frombuffer(r2,'<f2',count=16*4096,offset=0xC0000).astype(np.float32).reshape(16,4096)
        qpos=(1,5,6,7,10,11);kpos=(2,0,3,4,8,9)
        bias_index=np.array([[_deposit(q,qpos)+_deposit(k,kpos) for k in range(64)] for q in range(64)])
        self.register_buffer('bias',torch.from_numpy(bias_raw[:,bias_index].copy()))
        self.register_buffer('scale',torch.from_numpy(np.frombuffer(r2,'<f4',count=16,offset=0xE0000).copy()))
        self.register_buffer('projection',torch.from_numpy(_decode_packed_qmma_projection(r3,0,512,512)))
        self.register_buffer('attn_skip',torch.from_numpy(np.frombuffer(r3,'<f2',count=512,offset=0x40000).astype(np.float32).copy()))
        q=np.concatenate([np.argsort(_P32)+32*g for g in range(16)]);self.register_buffer('channel_route',torch.from_numpy(q))
        q64=np.concatenate([np.argsort(_P32),np.argsort(_P32)+32]);self.register_buffer('factor_input_route',torch.from_numpy(q64))
        phase_base=40 if 40<=block<=47 else 23
        self.shift=((0,0),(4,4),(4,0),(0,4))[(block-phase_base)%4]
        if block==30:
            r4=_raw(root,block,4)
            self.register_buffer('terminal',torch.from_numpy(_decode_packed_qmma_projection(r4,0,512,1024)))
        else:self.terminal=None
    @staticmethod
    def _partition(x):
        b,h,w,heads,d=x.shape
        return x.reshape(b,h//8,8,w//8,8,heads,d).permute(0,1,3,5,2,4,6).reshape(-1,heads,64,d)
    @staticmethod
    def _reverse(x,b,h,w):
        return x.reshape(b,h//8,w//8,16,8,8,32).permute(0,1,4,2,5,3,6).reshape(b,h,w,512)
    def _ffn_expand(self,x):
        if not self.factorized_expand:
            source=x.index_select(-1,self.channel_route)
            gate=matmul_fp16_linear(source, self.gate);value=matmul_fp16_linear(source, self.value)
            return quantize_e4m3_2h(value*fast_activation_2h(gate))
        # SM89 seq101: one 512-wide input projection, E4M3 boundary, then
        # four 64-wide factor paths.  Activation is between the two factor
        # matrices and their FP16 outputs accumulate before the final E4M3 store.
        source=x.index_select(-1,self.channel_route)
        projected=(matmul_fp16_linear(source.to(torch.float16), self.expand_input.to(torch.float16))).to(torch.float16)
        projected=quantize_e4m3_2h(projected)
        grouped=projected.reshape(*projected.shape[:-1],8,64).index_select(-1,self.factor_input_route)
        output=torch.zeros_like(grouped,dtype=torch.float16)
        for rep in range(4):
            hidden=matmul_fp16_linear(grouped.unsqueeze(-2),self.expand_factor_in[rep]).squeeze(-2)
            hidden=quantize_e4m3_2h(fast_activation_2h(hidden))
            # Native grouped K64 contraction keeps C across all four reps.
            # block40-ffn0-sm89 has no HADD2 after these projections: the
            # previous rep is the initial accumulator, not a separate sum.
            output=matmul_fp16_linear(hidden.unsqueeze(-2),self.expand_factor_out[rep],
                                      initial=output.unsqueeze(-2)).squeeze(-2)
        return quantize_e4m3_2h(output.float().reshape(*x.shape[:-1],512))

    def _ffn_project(self,hidden,skip):
        source=hidden.index_select(-1,self.channel_route)
        residual=skip.half()*self.ffn_skip.half()
        projected=matmul_fp16_linear(source, self.ffn_projection, initial=residual)
        return quantize_e4m3_2h(projected)

    def _attention_project(self,attended,skip,*,quantize_output=True):
        source=attended.index_select(-1,self.channel_route)
        residual=skip.half()*self.attn_skip.half()
        projected=matmul_fp16_linear(source, self.projection, initial=residual)
        return quantize_e4m3_2h(projected) if quantize_output else projected.half().float()

    @staticmethod
    def _slot_from_xy(y,x):
        return ((y&1)<<0)|(((y>>1)&1)<<1)|((x&1)<<2)|(((x>>1)&1)<<3)|(((y>>2)&1)<<4)|(((x>>2)&1)<<5)

    def _attention_entries(self,oh,ow):
        """Map the SM89 fused spatial index to shifted 8x8 slots/regions."""
        hp=(oh+7)&-8;wp=(ow+7)&-8;sy,sx=self.shift;groups={}
        for s in range(oh*ow):
            wx=s//(oh*8);rem=s%(oh*8);side=rem//(oh*4);within=rem%(oh*4);wy=within//32;slot=within%32+side*32
            ly=((slot>>0)&1)|(((slot>>1)&1)<<1)|(((slot>>4)&1)<<2)
            lx=((slot>>2)&1)|(((slot>>3)&1)<<1)|(((slot>>5)&1)<<2)
            y=wy*8+ly;x=wx*8+lx
            if y>=oh or x>=ow:continue
            ys=(y-sy)%hp;xs=(x-sx)%wp
            region_y=(0 if sy==0 else (0 if ys<hp-8 else (1 if ys<hp-4 else 2)))
            region_x=(0 if sx==0 else (0 if xs<wp-8 else (1 if xs<wp-4 else 2)))
            key=(ys//8,xs//8,region_y,region_x)
            groups.setdefault(key,[]).append((self._slot_from_xy(ys%8,xs%8),s))
        return list(groups.values())

    def _attention_hwc(self,x,oh,ow):
        """Encoder inpview variant: ordinary HWC coordinates, Morton slots."""
        b=x.shape[0];valid=x[:,:oh,:ow].reshape(b,oh*ow,512)
        z=(matmul_fp16_linear(valid.index_select(-1, self.channel_route), self.qkv)).reshape(b,oh*ow,16,96)
        q,k,v=(z[...,i*32:(i+1)*32] for i in range(3))
        q=quantize_e4m3_2h(normalize_fp16_2h(q)*self.scale.half().reshape(1,1,16,1))
        k=quantize_e4m3_2h(normalize_fp16_2h(k));v=quantize_e4m3_2h(v)
        out=torch.zeros((b,oh*ow,16,32),device=x.device,dtype=x.dtype)
        hp=(oh+7)&-8;wp=(ow+7)&-8
        phase=((0,0),(4,4),(4,0),(0,4))[(self.block-23)%4];sy,sx=phase;groups={}
        for y in range(oh):
            for xx in range(ow):
                ys=(y-sy)%hp;xs=(xx-sx)%wp
                ry=(0 if sy==0 else (0 if ys<hp-8 else (1 if ys<hp-4 else 2)))
                rx=(0 if sx==0 else (0 if xs<wp-8 else (1 if xs<wp-4 else 2)))
                yy,xxx=ys%8,xs%8
                bits=[(yy>>i)&1 for i in range(3)]+[(xxx>>i)&1 for i in range(3)]
                special=sum(bits[i]<<p for i,p in enumerate((3,0,4,1,2,5)))
                groups.setdefault((ys//8,xs//8,ry,rx),[]).append((self._slot_from_xy(yy,xxx),special,y*ow+xx))
        for entries in groups.values():
            slots=[u for _,u,_ in entries];ids=[s for _,_,s in entries]
            Q=torch.zeros((b,16,64,32),device=x.device,dtype=x.dtype);K=torch.zeros_like(Q);V=torch.zeros_like(Q)
            Q[:,:,slots]=q[:,ids].permute(0,2,1,3);K[:,:,slots]=k[:,ids].permute(0,2,1,3);V[:,:,slots]=v[:,ids].permute(0,2,1,3)
            logits=matmul_fp16_accumulate(Q,K.transpose(-1,-2),initial=self.bias.half().unsqueeze(0))
            affine = affine_exp_input_2h(logits)
            weight=fast_exp_2h(affine);probability=quantize_e4m3_2h(normalize_sum_fp16(weight, denominator=sum_fp16_key64(weight, "16h")))
            attended=quantize_e4m3_2h(matmul_fp16_accumulate(probability.to(torch.float16),V.to(torch.float16)));out[:,ids]=attended[:,:,slots].permute(0,2,1,3)
        return out.reshape(b,oh,ow,512)

    def _attention_physical(self,x,oh,ow):
        b=x.shape[0];valid=x[:,:oh,:ow].reshape(b,oh*ow,512)
        z=(matmul_fp16_linear(valid.index_select(-1, self.channel_route), self.qkv)).reshape(b,oh*ow,16,96)
        q,k,v=(z[...,i*32:(i+1)*32] for i in range(3))
        q=quantize_e4m3_2h(normalize_fp16_2h(q)*self.scale.half().reshape(1,1,16,1))
        k=quantize_e4m3_2h(normalize_fp16_2h(k));v=quantize_e4m3_2h(v)
        out=torch.zeros((b,oh*ow,16,32),device=x.device,dtype=x.dtype)
        for entries in self._attention_entries(oh,ow):
            slots=[u for u,_ in entries];ids=[s for _,s in entries]
            Q=torch.zeros((b,16,64,32),device=x.device,dtype=x.dtype);K=torch.zeros_like(Q);V=torch.zeros_like(Q)
            Q[:,:,slots]=q[:,ids].permute(0,2,1,3);K[:,:,slots]=k[:,ids].permute(0,2,1,3);V[:,:,slots]=v[:,ids].permute(0,2,1,3)
            logits=matmul_fp16_accumulate(Q,K.transpose(-1,-2),initial=self.bias.half().unsqueeze(0))
            affine = affine_exp_input_2h(logits)
            weight=fast_exp_2h(affine)
            probability=quantize_e4m3_2h(normalize_sum_fp16(weight, denominator=sum_fp16_key64(weight, "16h")))
            attended=quantize_e4m3_2h(matmul_fp16_accumulate(probability.to(torch.float16),V.to(torch.float16)))
            out[:,ids]=attended[:,:,slots].permute(0,2,1,3)
        return out.reshape(b,oh,ow,512)

    def forward(self,feature,*,quantize_output=True):
        b,oh,ow,c=feature.shape
        if c!=512:raise ValueError(c)
        h=(oh+7)&-8;w=(ow+7)&-8
        if (h,w)!=(oh,ow):feature=F.pad(feature.permute(0,3,1,2),(0,w-ow,0,h-oh)).permute(0,2,3,1)
        x=quantize_e4m3_2h(feature)
        hidden=self._ffn_expand(x)
        x=self._ffn_project(hidden,x)
        if 23<=self.block<=30:
            attended=self._attention_hwc(x,oh,ow)
        elif 40<=self.block<=47:
            attended=self._attention_physical(x,oh,ow)
        else:
            raise ValueError(f'unsupported physical512 block {self.block}')
        if (h,w)!=(oh,ow):attended=F.pad(attended.permute(0,3,1,2),(0,w-ow,0,h-oh)).permute(0,2,3,1)
        y=self._attention_project(attended,x,quantize_output=quantize_output)
        return y[:,:oh,:ow]
    def terminal_downsample(self,projection_half):
        if self.terminal is None:raise RuntimeError('not a terminal split-Swin block')
        # proj_pool publishes an FP8 skip, but its SHFL/HADD2 pool still reads
        # the original projection accumulators. Both native inputs and all
        #32768 small-image pooled bytes confirm this prequant half-pair tree.
        skip=quantize_e4m3_2h(projection_half)
        source=projection_half.half()
        top=source[:,0::2,0::2]+source[:,0::2,1::2]
        bottom=source[:,1::2,0::2]+source[:,1::2,1::2]
        pooled=((top+bottom)*0.25).float()
        # Native seq56 publishes a 4-aligned descriptor: 20x16 -> 12x8,
        # not 10x8. All96 rows then pass through the actual ViT kernels.
        h,w=pooled.shape[1:3];ph=(h+3)&-4;pw=(w+3)&-4
        if (ph,pw)!=(h,w):pooled=F.pad(pooled.permute(0,3,1,2),(0,pw-w,0,ph-h)).permute(0,2,3,1)
        pooled=quantize_e4m3_2h(pooled).index_select(-1,self.channel_route)
        return quantize_e4m3_2h(matmul_fp16_linear(pooled, self.terminal)),skip

__all__=['TorchPhysicalSplitSwin16H']


"""Caller-owned resources and geometry for the pure-Torch block70 compositor."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Block70RuntimeInputs:
    """Host inputs recovered from IDA's DLSSNR Evaluate→block70 chain.

    Image resources may be NHWC or NCHW. Rectangles are `[baseX,baseY,W,H]`;
    omitted rectangles mean the full resource. `mvec_scale_xy` is the caller's
    DLSSNR.MVecScaleX/Y.  The normalized +0xa4/+0xa8 launch values are obtained
    by dividing by the active Color width/height.
    """

    color: torch.Tensor
    prev_output: torch.Tensor
    mvec: torch.Tensor
    mvec_scale_xy: torch.Tensor
    output_dimensions_wh: torch.Tensor | None = None
    color_rect_xywh: torch.Tensor | None = None
    prev_rect_xywh: torch.Tensor | None = None
    mvec_rect_xywh: torch.Tensor | None = None
    rgb_mode: bool = True
    intensity: float = 1.0
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    use_auto_mask: bool = True
    style: int = 0
    backbuffer: torch.Tensor | None = None
    backbuffer_rect_xywh: torch.Tensor | None = None
    control_mask: torch.Tensor | None = None
    control_mask_rect_xywh: torch.Tensor | None = None

    @staticmethod
    def _channels(value: torch.Tensor) -> int:
        if value.ndim != 4:
            return -1
        if value.shape[-1] <= 4:
            return int(value.shape[-1])
        if value.shape[1] <= 4:
            return int(value.shape[1])
        return -1

    def _color_size(self) -> tuple[int, int]:
        if self.color.shape[-1] <= 4:
            return int(self.color.shape[2]), int(self.color.shape[1])
        return int(self.color.shape[3]), int(self.color.shape[2])

    def normalized_mvec_scale(self, width: int, height: int) -> torch.Tensor:
        values = self.mvec_scale_xy.detach().reshape(-1).to(torch.float32)
        if values.numel() != 2:
            raise ValueError("block70 MVecScaleX/Y requires two values")
        if self.color_rect_xywh is None:
            color_width, color_height = self._color_size()
        else:
            rect = self.color_rect_xywh.detach().reshape(-1).to(torch.float32)
            if rect.numel() != 4:
                raise ValueError("block70 Color rect must be [x,y,width,height]")
            color_width, color_height = float(rect[2]), float(rect[3])
        return values / torch.tensor(
            [float(color_width or width), float(color_height or height)],
            dtype=torch.float32,
            device=values.device,
        )

    def validate(self, batch: int, height: int, width: int) -> None:
        for name, value, minimum_channels in (
            ("color", self.color, 3),
            ("prev_output", self.prev_output, 3),
            ("mvec", self.mvec, 2),
        ):
            channels = self._channels(value)
            if channels < minimum_channels or value.shape[0] not in (1, batch):
                raise ValueError(
                    f"block70 {name}: expected batch-compatible rank-4 tensor "
                    f"with at least {minimum_channels} channels"
                )
        for name, rect in (
            ("color", self.color_rect_xywh),
            ("prev", self.prev_rect_xywh),
            ("mvec", self.mvec_rect_xywh),
            ("backbuffer", self.backbuffer_rect_xywh),
            ("control_mask", self.control_mask_rect_xywh),
        ):
            if rect is not None and rect.numel() != 4:
                raise ValueError(f"block70 {name} rect must be [x,y,width,height]")
        if self.output_dimensions_wh is not None:
            dimensions = self.output_dimensions_wh.detach().reshape(-1).to(torch.float32)
            if dimensions.numel() != 2:
                raise ValueError("block70 output dimensions require width,height")
            if int(dimensions[0].item()) != width or int(dimensions[1].item()) != height:
                raise ValueError(
                    f"block70 output dimensions say {tuple(dimensions.tolist())}, "
                    f"but network output is {(width, height)}"
                )
        if not torch.isfinite(torch.tensor(float(self.intensity))):
            raise ValueError("block70 intensity must be finite")
        for name, value in (
            ("local_tone_strength", self.local_tone_strength),
            ("local_structure_strength", self.local_structure_strength),
            ("skin_structure_strength", self.skin_structure_strength),
        ):
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")
        if not 0 <= int(self.style) <= 255:
            raise ValueError("style must be an unsigned byte")
        if self.backbuffer is not None and (
            self._channels(self.backbuffer) < 3 or self.backbuffer.shape[0] not in (1, batch)
        ):
            raise ValueError("block70 backbuffer requires batch-compatible RGB")
        if self.control_mask is not None and (
            self._channels(self.control_mask) < 1 or self.control_mask.shape[0] not in (1, batch)
        ):
            raise ValueError("block70 control_mask requires a batch-compatible channel")


__all__ = ["Block70RuntimeInputs"]


"""Pure-Torch translation of the block70 color/history compositor.

The high-level equations and signs come from block70-post.sass.  CUDA texture
instructions are represented with PyTorch sampling over caller-provided image
tensors; instruction-level interpolation rounding is intentionally not emulated.
"""

import torch
import torch.nn.functional as F



def _nchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"expected rank-4 image, got {tuple(image.shape)}")
    # Runtime resources are documented NHWC.  Accept NCHW when unambiguous.
    if image.shape[-1] <= 4:
        return image.permute(0, 3, 1, 2)
    if image.shape[1] <= 4:
        return image
    raise ValueError(f"cannot identify image channels: {tuple(image.shape)}")


def _nhwc(image: torch.Tensor) -> torch.Tensor:
    return _nchw(image).permute(0, 2, 3, 1)


def _full_rect(image: torch.Tensor) -> torch.Tensor:
    nchw = _nchw(image)
    return torch.tensor(
        [0.0, 0.0, float(nchw.shape[3]), float(nchw.shape[2])],
        device=image.device,
        dtype=torch.float32,
    )


def _output_uv(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def _rect_uv(output_uv: torch.Tensor, image: torch.Tensor, rect: torch.Tensor | None) -> torch.Tensor:
    rect = _full_rect(image) if rect is None else rect.to(output_uv.device, torch.float32).reshape(4)
    nchw = _nchw(image)
    resource_size = torch.tensor(
        [float(nchw.shape[3]), float(nchw.shape[2])],
        device=output_uv.device,
        dtype=torch.float32,
    )
    return (rect[:2] + output_uv * rect[2:]) / resource_size


def bilinear_texture(image: torch.Tensor, uv: torch.Tensor, *, border: bool = False) -> torch.Tensor:
    """CUDA-like normalized 2D linear texture lookup at pixel-center UVs."""
    source = _nchw(image).to(torch.float32)
    if source.shape[0] == 1 and uv.shape[0] != 1:
        source = source.expand(uv.shape[0], -1, -1, -1)
    grid = uv.mul(2.0).sub(1.0)
    sampled = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border" if border else "zeros",
        align_corners=False,
    )
    return sampled.permute(0, 2, 3, 1)


def catmull_rom_weights(t: torch.Tensor) -> torch.Tensor:
    """Four exact Catmull-Rom weights visible at SASS 0xa2d0..0xa4f0."""
    t2 = t * t
    t3 = t2 * t
    return torch.stack(
        (
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ),
        dim=-1,
    )


def catmull_rom_texture(
    image: torch.Tensor,
    uv: torch.Tensor,
    *,
    output_size: tuple[int, int] | None = None,
    rect: torch.Tensor | None = None,
) -> torch.Tensor:
    """Five-bilinear-tap normalized Catmull-Rom cross used by the CUBIN.

    This is not a naïve discrete 4x4 convolution. The kernel collapses the
    central 2x2 support into one bilinear tap, keeps four axial outer taps,
    omits corner terms, and normalizes by the retained-weight sum.
    """
    source = _nhwc(image).to(torch.float32)
    batch, src_h, src_w, _ = source.shape
    out_h, out_w = output_size or (src_h, src_w)
    sx = uv[..., 0] * out_w - 0.5
    sy = uv[..., 1] * out_h - 0.5
    ix = torch.floor(sx)
    iy = torch.floor(sy)
    tx = torch.clamp(uv[..., 0] * out_w - (ix + 0.5), 0.0, 1.0)
    ty = torch.clamp(uv[..., 1] * out_h - (iy + 0.5), 0.0, 1.0)
    wx = catmull_rom_weights(tx)
    wy = catmull_rom_weights(ty)
    gx = wx[..., 1] + wx[..., 2]
    gy = wy[..., 1] + wy[..., 2]
    eps = torch.finfo(torch.float32).eps

    x_left = torch.clamp(ix - 0.5, 0.5, out_w - 0.5)
    x_center = torch.clamp(ix + 0.5 + wx[..., 2] / gx.clamp_min(eps), 0.5, out_w - 0.5)
    x_right = torch.clamp(ix + 2.5, 0.5, out_w - 0.5)
    y_top = torch.clamp(iy - 0.5, 0.5, out_h - 0.5)
    y_center = torch.clamp(iy + 0.5 + wy[..., 2] / gy.clamp_min(eps), 0.5, out_h - 0.5)
    y_bottom = torch.clamp(iy + 2.5, 0.5, out_h - 0.5)

    def mapped(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        normalized = torch.stack((x / out_w, y / out_h), dim=-1)
        return _rect_uv(normalized, image, rect)

    top = bilinear_texture(image, mapped(x_center, y_top), border=True)
    left = bilinear_texture(image, mapped(x_left, y_center), border=True)
    center = bilinear_texture(image, mapped(x_center, y_center), border=True)
    bottom = bilinear_texture(image, mapped(x_center, y_bottom), border=True)
    right = bilinear_texture(image, mapped(x_right, y_center), border=True)

    wt = gx * wy[..., 0]
    wl = wx[..., 0] * gy
    wc = gx * gy
    wb = gx * wy[..., 3]
    wr = wx[..., 3] * gy
    denominator = (wt + wl + wc + wb + wr).clamp_min(eps)
    return (
        top * wt.unsqueeze(-1)
        + left * wl.unsqueeze(-1)
        + center * wc.unsqueeze(-1)
        + bottom * wb.unsqueeze(-1)
        + right * wr.unsqueeze(-1)
    ) / denominator.unsqueeze(-1)


def reconstruct_block70_color(
    residual_rgb: torch.Tensor,
    history_logit: torch.Tensor,
    runtime: Block70RuntimeInputs,
    blend_scale: torch.Tensor,
) -> torch.Tensor:
    """Translate the common RGB/simple-blend path at SASS 0x9d60..0xab10.

    Equations:
      corrected_centered = (Color - 0.5)/8 + residual_rgb/32
      corrected_rgb      = sat(corrected_centered*8 + 0.5)  [RGB variant]
      history_uv         = output_uv + MVec * normalized_mvec_scale
      alpha              = sat(blend_scale) * sigmoid(history_logit)
      output             = corrected + alpha*(history-corrected)
    """
    if residual_rgb.ndim != 4 or residual_rgb.shape[-1] != 3:
        raise ValueError("block70 residual_rgb must be NHWC RGB")
    batch, height, width, _ = residual_rgb.shape
    runtime.validate(batch, height, width)
    device = residual_rgb.device
    uv = _output_uv(batch, height, width, device)

    color_uv = _rect_uv(uv, runtime.color, runtime.color_rect_xywh)
    current = bilinear_texture(runtime.color.to(device), color_uv)[..., :3]
    centered = current * 0.125 - 0.0625
    corrected = centered + residual_rgb.to(torch.float32) * 0.03125
    if runtime.rgb_mode:
        corrected = torch.clamp(corrected * 8.0 + 0.5, 0.0, 1.0)

    if runtime.rgb_mode:
        mvec_uv = _rect_uv(uv, runtime.mvec, runtime.mvec_rect_xywh)
        motion = bilinear_texture(runtime.mvec.to(device), mvec_uv)[..., :2]
        scale = runtime.normalized_mvec_scale(width, height).to(device).reshape(1, 1, 1, 2)
        history_uv = uv + motion * scale
        history = catmull_rom_texture(
            runtime.prev_output.to(device),
            history_uv,
            output_size=(height, width),
            rect=runtime.prev_rect_xywh,
        )[..., :3]
        alpha = torch.sigmoid(history_logit.to(torch.float32))
        alpha = alpha * torch.clamp(blend_scale.to(device, torch.float32), 0.0, 1.0)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(-1)
        output = corrected + alpha * (history - corrected)
    else:
        output = corrected

    composite_requested = runtime.backbuffer is not None or runtime.control_mask is not None or runtime.intensity != 1.0
    if composite_requested:
        if runtime.backbuffer is None:
            raise ValueError("block70 output-composite mode requires DLSSNR.Backbuffer")
        backbuffer_resource = runtime.backbuffer
        backbuffer_uv = _rect_uv(uv, backbuffer_resource, runtime.backbuffer_rect_xywh)
        backbuffer = bilinear_texture(backbuffer_resource.to(device), backbuffer_uv)[..., :3]
        strength = torch.full_like(output[..., :1], float(runtime.intensity))
        if runtime.control_mask is not None:
            mask_uv = _rect_uv(uv, runtime.control_mask, runtime.control_mask_rect_xywh)
            mask = bilinear_texture(runtime.control_mask.to(device), mask_uv)[..., :1]
            # Control-mask SASS 0xac10 saturates the product, not each
            # operand independently: effective = sat(Intensity * sampledMask).
            strength = strength * mask
        strength = torch.clamp(strength, 0.0, 1.0)
        output = torch.clamp(backbuffer + strength * (output - backbuffer), 0.0, 1.0)

    return output


__all__ = [
    "bilinear_texture",
    "catmull_rom_texture",
    "catmull_rom_weights",
    "reconstruct_block70_color",
]


"""Pure-Torch construction of the block0 pre-HMMA packet."""

import math
import torch


_U32 = 0xFFFFFFFF


def pad_color_for_neural_buffer(color: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Extend BHWC Color using block0's single reflection then sampler clamp.

    This is not periodic reflect-padding. SM89 0x0240–0x02d0 and
    0x03d0–0x0400 choose coord or 2*size-2-coord once. Negative reflected
    coordinates are then clamped by the Color sampler.
    """
    if color.ndim != 4 or color.shape[-1] not in (3, 4):
        raise ValueError("Color must be BHWC RGB/RGBA")
    h, w = color.shape[1:3]
    if not h or not w or height < h or width < w:
        raise ValueError((tuple(color.shape), height, width))
    y = torch.arange(height, device=color.device)
    x = torch.arange(width, device=color.device)
    y = torch.where(y < h, y, 2*h-2-y).clamp(0, h-1)
    x = torch.where(x < w, x, 2*w-2-x).clamp(0, w-1)
    return color.index_select(1, y).index_select(2, x)


def _u32(value: torch.Tensor) -> torch.Tensor:
    return value & _U32


def _pcg_r(value: torch.Tensor) -> torch.Tensor:
    value = _u32(value)
    shift = (value >> 28) + 4
    return _u32(((value >> shift) ^ value) * 0x108EF2D9)


def _pcg_output(value: torch.Tensor) -> torch.Tensor:
    r = _pcg_r(value)
    return _u32((r >> 22) ^ r)


def _uniform24(value: torch.Tensor) -> torch.Tensor:
    r = _pcg_r(value)
    integer = _u32((r >> 30) ^ (r >> 8)) + 1
    return integer.to(torch.float32) * (2.0 ** -24)


def gaussian_dither(
    height: int,
    width: int,
    *,
    frame: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the three deterministic Box-Muller values stored by block0.

    This translates the PCG-style coordinate/frame hash at pre-block SASS
    0x03d0..0x0ad0.  The result is ``[1,H,W,3]`` ordered ``g0,g1,g2``.
    """
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.int64, device=device),
        torch.arange(width, dtype=torch.int64, device=device),
        indexing="ij",
    )
    q = _u32(
        _u32(x * 0x8DA6B343)
        ^ _u32(y * 0xD8163841)
        ^ _u32(torch.full_like(x, int(frame)) * 0x9E3779B9)
        ^ 0x243F6A88
    )
    h = _pcg_output(q)
    states = (
        _u32(h * 0xCAA5B80D + 0x21DD796B),
        _u32(h * 0x83232C31 + 0x3463E0AC),
        _u32(h * 0x2C9277B5 + 0xAC564B05),
        _u32(h * 0xFA6DC5F9 + 0x4712A88E),
    )
    u0, u1, u2, u3 = (_uniform24(state) for state in states)
    radius0 = torch.sqrt(-2.0 * torch.log(u0))
    radius1 = torch.sqrt(-2.0 * torch.log(u2))
    angle0 = (2.0 * math.pi) * u1
    angle1 = (2.0 * math.pi) * u3
    g0 = radius0 * torch.cos(angle0)
    g1 = radius1 * torch.cos(angle1)
    g2 = radius1 * torch.sin(angle1)
    # The CUBIN rounds each value to FP16 before writing shared memory.
    return torch.stack((g0, g1, g2), dim=-1).to(torch.float16).to(torch.float32).unsqueeze(0)


def build_preblock_features(
    runtime: Block70RuntimeInputs,
    *,
    height: int | None = None,
    width: int | None = None,
    color_scale: float | None = None,
    style: int | None = None,
    frame: int = 0,
) -> torch.Tensor:
    """Build the exact no-ControlMask FP16 packet consumed by block0 HMMA.

    The returned BCHW tensor has sixteen channels in shared-store order::

      g1,g2,g0,1,current.R,current.G,current.B,source.R,
      source.G,source.B,style/128,tone,structure,skin,auto,0

    Color/source components are centered and multiplied in half precision by
    ``2*c[0][0x224]`` in runtime SM89 (parameter base 0x160).
    Captured qword24.high is 0.0625, hence the effective scale is 0.125.  ``source`` is
    current Color unless both MVec and PrevOutput are bound, in which case it
    is the positively warped five-tap Catmull-Rom history sample.
    """
    batch = int(runtime.color.shape[0])
    if runtime.output_dimensions_wh is not None:
        dimensions = runtime.output_dimensions_wh.detach().reshape(-1)
        width = int(dimensions[0].item()) if width is None else width
        height = int(dimensions[1].item()) if height is None else height
    if runtime.color.shape[-1] <= 4:
        default_height, default_width = int(runtime.color.shape[1]), int(runtime.color.shape[2])
    else:
        default_height, default_width = int(runtime.color.shape[2]), int(runtime.color.shape[3])
    height = default_height if height is None else int(height)
    width = default_width if width is None else int(width)
    runtime.validate(batch, height, width)

    device = runtime.color.device
    uv = _output_uv(batch, height, width, device)
    color_uv = _rect_uv(uv, runtime.color, runtime.color_rect_xywh)
    current = bilinear_texture(runtime.color, color_uv)[..., :3]

    mvec_uv = _rect_uv(uv, runtime.mvec, runtime.mvec_rect_xywh)
    motion = bilinear_texture(runtime.mvec, mvec_uv)[..., :2]
    normalized_scale = runtime.normalized_mvec_scale(width, height).to(device).reshape(1, 1, 1, 2)
    history_uv = uv + motion * normalized_scale
    source = catmull_rom_texture(
        runtime.prev_output,
        history_uv,
        output_size=(height, width),
        rect=runtime.prev_rect_xywh,
    )[..., :3]

    sample_scale = torch.tensor(
        0.125 if color_scale is None else float(color_scale),
        device=device,
        dtype=torch.float16,
    )
    half_center = torch.tensor(0.5, device=device, dtype=torch.float16)
    current_centered = (
        (current.to(torch.float16) - half_center) * sample_scale
    ).to(torch.float32)
    source_centered = (
        (source.to(torch.float16) - half_center) * sample_scale
    ).to(torch.float32)
    scalar_shape = (batch, height, width, 1)

    def scalar(value: float) -> torch.Tensor:
        return torch.full(scalar_shape, float(value), device=device, dtype=torch.float32)

    local_structure = float(runtime.local_structure_strength)
    if runtime.use_auto_mask:
        skin = float(runtime.skin_structure_strength)
        effective_skin = skin if skin >= 0.0 else local_structure
        effective_auto = local_structure
        both_masks = effective_skin >= 0.0 and effective_auto >= 0.0
        structure_slot = 1.0 if both_masks else local_structure
        skin_slot = -1.0 if both_masks else effective_skin
        auto_slot = -1.0 if both_masks else effective_auto
    else:
        # Runtime payload qword23 is exactly (-1,-1) when Automatic Mask is
        # disabled.  The previous fallback silently replaced both sentinels
        # with local_structure, changing the block0 learned operating point.
        structure_slot = local_structure
        skin_slot = -1.0
        auto_slot = -1.0

    gaussian = gaussian_dither(height, width, frame=frame, device=device).expand(batch, -1, -1, -1)
    packet = torch.cat(
        (
            gaussian[..., 1:3],
            gaussian[..., 0:1],
            scalar(1.0),
            current_centered,
            source_centered[..., 0:1],
            source_centered[..., 1:3],
            scalar(float(runtime.style if style is None else style) * (1.0 / 128.0)),
            scalar(runtime.local_tone_strength),
            scalar(structure_slot),
            scalar(skin_slot),
            scalar(auto_slot),
            scalar(0.0),
        ),
        dim=-1,
    )
    if packet.shape[-1] != 16:
        raise AssertionError(packet.shape)
    return packet.permute(0, 3, 1, 2).contiguous()


__all__ = ["build_preblock_features", "gaussian_dither"]


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARENA = DEFAULT_ROOT
WINDOW_SIZE = 8
SHIFT_SIZE = 4

# Four spatial phases are separate CUBIN kernels: no shift, XY, X-only,
# Y-only. The old boolean table collapsed the final three into XY.
SWIN_PHASES = ((0, 0), (4, 4), (0, 4), (4, 0))
ENCODER_SHIFTED = {
    0: (0, 0),
    **{block: SWIN_PHASES[(block - 1) % 4] for block in range(1, 5)},
    **{block: SWIN_PHASES[(block - 5) % 4] for block in range(5, 9)},
    **{block: SWIN_PHASES[(block - 9) % 4] for block in range(9, 15)},
    **{block: SWIN_PHASES[(block - 15) % 4] for block in range(15, 23)},
}


@dataclass(frozen=True)
class BlockTrace:
    block: int
    shape: tuple[int, ...]
    finite: bool
    absmax: float
    mean: float
    std: float
    confidence: str


@dataclass(frozen=True)
class EncoderResult:
    """Encoder output0 from block22 plus output1 values used by decoder skips."""

    main: torch.Tensor
    skips: dict[int, torch.Tensor]
    trace: tuple[BlockTrace, ...]


class ExportedWeights:
    """Name-indexed access to ``weights/*.npy`` without rescanning the manifest."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.directory = root / "weights"
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("count") != 153:
            raise ValueError(f"expected 153 records, got {manifest.get('count')}")
        self._records = {item["name"]: item for item in manifest["tensors"]}

    def numpy(self, name: str) -> np.ndarray:
        item = self._records[name]
        value = np.load(self.directory / item["file"])
        if value.size != int(item["element_count"]):
            raise ValueError(
                f"{name}: manifest says {item['element_count']} elements, file has {value.size}"
            )
        return value.astype(np.float32, copy=False)

    def tensor(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.numpy(name).copy())


class RecoveredGraph:
    """Validate and expose the authoritative CPU-descriptor graph."""

    def __init__(self, arena: Path = DEFAULT_ARENA) -> None:
        self.path = arena / "network-graph.json"
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.blocks: list[dict[str, object]] = raw["blocks"]
        self.sample_sha256 = raw["sample_sha256"]
        self.config = raw["config"]
        self.architecture_variant = raw["architecture_variant"]
        self._validate()

    def _validate(self) -> None:
        ids = [int(item["block"]) for item in self.blocks]
        if ids != list(range(71)):
            raise ValueError("network graph must contain contiguous blocks 0..70")
        records = [
            name
            for block in self.blocks
            for name in block.get("weight_records", [])  # type: ignore[union-attr]
        ]
        if len(records) != 153 or len(set(records)) != 153:
            raise ValueError(f"expected 153 unique graph weight records, got {len(records)}")
        skip_edges = []
        for block in self.blocks:
            inputs = block.get("inputs") or []
            if len(inputs) == 2:  # type: ignore[arg-type]
                skip_edges.append(int(block["block"]))
        if skip_edges != [39, 48, 56, 62, 66, 70]:
            raise ValueError(f"unexpected two-input blocks: {skip_edges}")

    @property
    def supported_blocks(self) -> range:
        return range(23)





def fast_activation(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(-4.0, 4.0)
    gate = 0.447265625 - 0.055908203125 * value.abs()
    gate = 0.89453125 + value * gate
    return value * gate


def _window_partition(value: torch.Tensor, window: int = WINDOW_SIZE) -> torch.Tensor:
    batch, height, width, heads, dim = value.shape
    return (
        value.reshape(batch, height // window, window, width // window, window, heads, dim)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(-1, heads, window * window, dim)
    )


def _window_reverse(
    windows: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    channels: int,
    window: int = WINDOW_SIZE,
) -> torch.Tensor:
    heads = channels // 32
    return (
        windows.reshape(batch, height // window, width // window, heads, window, window, 16)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .reshape(batch, height, width, channels // 2)
    )


def _shift_region_mask(
    height: int,
    width: int,
    device: torch.device,
    shift: tuple[int, int] = (SHIFT_SIZE, SHIFT_SIZE),
) -> torch.Tensor:
    """8x8 connectivity mask for independent Y/X shift phases."""

    mask = torch.zeros((1, height, width, 1), device=device)
    shift_y, shift_x = shift
    h_slices = (
        (slice(0, -8), slice(-8, -shift_y), slice(-shift_y, None))
        if shift_y else (slice(None),)
    )
    w_slices = (
        (slice(0, -8), slice(-8, -shift_x), slice(-shift_x, None))
        if shift_x else (slice(None),)
    )
    region = 0
    for hs in h_slices:
        for ws in w_slices:
            mask[:, hs, ws, :] = region
            region += 1
    windows = (
        mask.reshape(1, height // 8, 8, width // 8, 8, 1)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, 64)
    )
    difference = windows.unsqueeze(1) - windows.unsqueeze(2)
    return difference.ne(0)




def publish_first_2h(feature: torch.Tensor) -> torch.Tensor:
    """Block5 native output publication into the next 2H consumer layout.

    The local block's algebra returns values in its input canonical layout.
    Its first-family output rearranges raw token/channel planes globally.
    This size-generic address formula equals all 1,310,720 recovered identity
    addresses; blocks6/7/8 publications are identities in this representation.
    """
    batch,height,width,channels=feature.shape
    if channels != 64 or height%8 or width%8:
        raise ValueError(tuple(feature.shape))
    tile=feature.reshape(batch,height//8,8,width//8,8,64).permute(0,1,3,2,4,5).reshape(batch,height//8,width//8,2,32,4,16)
    raw=tile.permute(0,5,1,2,3,4,6).reshape(batch,-1)
    d=torch.arange(height*width*64,device=feature.device,dtype=torch.int64)
    outer=d>>10
    source=((outer//(width//4))*(width*64)+(outer%(width//4))*64
            +(d&3)+((d>>4)&15)*4+((d>>2)&1)*width*32
            +((d>>3)&1)*height*width*16+((d>>8)&1)*width*16
            +((d>>9)&1)*height*width*32)
    published=raw[:,source].reshape(batch,4,height//8,width//8,2,32,16)
    tile=published.permute(0,2,3,4,5,1,6).reshape(batch,height//8,width//8,8,8,64)
    return tile.permute(0,1,3,2,4,5).reshape(batch,height,width,64)


class InputBlock(nn.Module):
    """Block0 pre-Swin plus its distinct full-resolution skip and compact chain output."""

    def __init__(self, archive: ExportedWeights, arena: Path = DEFAULT_ARENA) -> None:
        super().__init__()
        values = archive.tensor("block0.layer0.layer")
        raw = values.detach().cpu().numpy().astype("<f2", copy=False)

        def hmma_b_fragment(byte_offset: int, register_pair: int) -> np.ndarray:
            # mma.m16n8k16 B fragment: lane owns four FP16 values at
            # k=(lane%4)*2+(slot%2)+8*(slot//2), n=lane//4.
            lane_values = np.frombuffer(
                raw.tobytes(), dtype="<f2", count=256, offset=byte_offset
            ).reshape(32, 8)[:, register_pair * 4 : register_pair * 4 + 4]
            matrix = np.empty((16, 8), dtype=np.float32)
            for lane in range(32):
                for slot in range(4):
                    k = (lane % 4) * 2 + (slot % 2) + 8 * (slot // 2)
                    n = lane // 4
                    matrix[k, n] = lane_values[lane, slot]
            return matrix

        # SM89 ds pre-block: the four B fragments are 0x2010/pair0,
        # 0x2010/pair1, 0x2210/pair0, 0x2210/pair1. F2FP interleaves
        # each pair of N8 fragments into the next QMMA's K32 operand.
        fragments = [hmma_b_fragment(offset, pair)
                     for offset in (0x2010, 0x2210) for pair in (0, 1)]
        q_to_n = np.array([0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15,
                          16,17,24,25,18,19,26,27,20,21,28,29,22,23,30,31])
        # Retain the FP16 projection result for both the initial residual and
        # the compact-output average. The full skip is separately quantized.
        self.swin = TorchPhysical1HBlock(archive.directory.parent, 0, quantize_output=False)
        inverse = self.swin.channel_inverse.cpu().numpy()
        adapter = np.concatenate(fragments, axis=1)[:, q_to_n][:, inverse]
        self.register_buffer("input_adapter_weight", torch.from_numpy(adapter.T.copy()))
        # Shared sampler pixel -> canonical TinLayout tile, recovered from
        # LDS/HMMA/F2FP provenance and independently verified by double-skip
        # controlled publication. See debug-static/trace_block0_adapter.py.
        canonical_to_pixel = np.array([
            0,16,1,17,2,18,3,19,8,24,9,25,10,26,11,27,
            32,48,33,49,34,50,35,51,40,56,41,57,42,58,43,59,
            36,52,37,53,38,54,39,55,44,60,45,61,46,62,47,63,
            4,20,5,21,6,22,7,23,12,28,13,29,14,30,15,31], dtype=np.int64)
        self.register_buffer("canonical_to_pixel", torch.from_numpy(canonical_to_pixel))
        self.register_buffer("pixel_to_canonical", torch.from_numpy(np.argsort(canonical_to_pixel)))

    @staticmethod
    def _route_pixels(value: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = value.shape
        windows = value.reshape(batch, height//8, 8, width//8, 8, channels)
        windows = windows.permute(0,1,3,2,4,5).reshape(-1,64,channels)[:,route]
        return windows.reshape(batch,height//8,width//8,8,8,channels).permute(0,1,3,2,4,5).reshape(batch,height,width,channels)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != 16:
            raise ValueError(f"block0 expects BCHW with sixteen pre-HMMA channels, got {tuple(inputs.shape)}")
        packet = inputs.permute(0, 2, 3, 1).to(torch.float16)
        adapted = (matmul_fp16_linear(packet, self.input_adapter_weight.T.to(torch.float16))).to(torch.float32)
        feature = self._route_pixels(adapted, self.canonical_to_pixel)
        full_half = self.swin(feature)
        skip = quantize_e4m3(full_half)
        pixels = self._route_pixels(full_half, self.pixel_to_canonical).to(torch.float16)
        # SM89 bbd0..bcd0: two half2 pair sums, then half2 sum and x0.25.
        top = pixels[:,0::2,0::2] + pixels[:,0::2,1::2]
        bottom = pixels[:,1::2,0::2] + pixels[:,1::2,1::2]
        pooled = ((top + bottom) * 0.25).to(torch.float32)
        main = self._route_pixels(quantize_e4m3(pooled), self.canonical_to_pixel)
        return main, skip




def quantize_e4m3(value: torch.Tensor) -> torch.Tensor:
    """SM89 F2FP.SATFINITE.E4M3.F16, expressed without native FP8 dependence."""

    value = value.to(torch.float16).to(torch.float32)
    sign = torch.where(value < 0, -1.0, 1.0)
    absolute = value.abs()
    subnormal = torch.round(absolute * 512.0) / 512.0
    safe = absolute.clamp_min(torch.finfo(value.dtype).tiny)
    exponent = torch.floor(torch.log2(safe)).clamp(-6.0, 8.0)
    mantissa = torch.round((safe / torch.exp2(exponent) - 1.0) * 8.0)
    carry = mantissa >= 8.0
    exponent = exponent + carry.to(exponent.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    normal = torch.exp2(exponent) * (1.0 + mantissa / 8.0)
    result = torch.where(absolute < 0.015625, subnormal, normal)
    result = sign * result.clamp_max(448.0)
    return torch.where(value == 0, torch.zeros_like(result), result)


def _unpack_qmma_b_e4m3(tile: bytes) -> np.ndarray:
    """Official PTX m16n8k32 B-fragment lane mapping for one 512-byte subtile."""

    if len(tile) != 512:
        raise ValueError(f"QMMA-B subtile must be 512 bytes, got {len(tile)}")
    source = np.frombuffer(tile, dtype=np.uint8).reshape(32, 16)
    output = np.empty((2, 32, 8), dtype=np.uint8)
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        for half in range(2):
            for index, value in enumerate(source[lane, half * 8 : half * 8 + 8]):
                row = thread * 4 + (index & 3) + (16 if index >= 4 else 0)
                output[half, row, group] = value
    return output


def _decode_e4m3_numpy(value: np.ndarray) -> np.ndarray:
    byte = value.astype(np.uint8)
    sign = np.where(byte & 0x80, -1.0, 1.0)
    exponent = (byte >> 3) & 15
    mantissa = byte & 7
    normal = sign * (1.0 + mantissa / 8.0) * np.exp2(exponent.astype(np.float32) - 7.0)
    subnormal = sign * (mantissa / 8.0) * (2.0**-6)
    result = np.where(exponent == 0, subnormal, normal)
    result = np.where((exponent == 15) & (mantissa == 7), np.nan, result)
    if np.isnan(result).any():
        raise ValueError("unexpected NaN E4M3 value in block4 transition")
    return result.astype(np.float32)


def _decode_packed_qmma_projection(
    raw: bytes, start: int, input_width: int, output_width: int
) -> np.ndarray:
    """Decode K/32 planes of paired PTX m16n8k32 N8 B fragments."""

    transition_bytes = input_width * output_width
    if start < 0 or start + transition_bytes > len(raw):
        raise ValueError(
            f"invalid packed projection {start}:{start + transition_bytes}/{len(raw)}"
        )
    planes: list[np.ndarray] = []
    plane_bytes = 32 * output_width
    for plane in range(input_width // 32):
        base = start + plane * plane_bytes
        fragments = [
            _unpack_qmma_b_e4m3(raw[offset : offset + 512])
            for offset in range(base, base + plane_bytes, 512)
        ]
        planes.append(
            np.concatenate(
                [fragment[half] for fragment in fragments for half in range(2)],
                axis=1,
            )
        )
    packed = np.concatenate(planes, axis=0)
    transition = _decode_e4m3_numpy(packed)
    if transition.shape != (input_width, output_width):
        raise AssertionError(transition.shape)
    return transition




class PackedEncoderDownsample(nn.Module):
    """Swin body plus statically recovered K->2K compact projection.

    The four encoder stage boundaries share one physical scheme.  The ordinary
    record ends with 16 padding bytes; the DS record replaces that tail with
    K/32 planes.  Each plane stores K32xN fragments, two N8 groups per 512-byte
    subtile, where N=2K.  Official PTX m16n8k32 lane mapping recovers the unique
    row-major Kx2K E4M3 matrix without executing sm_120 code.
    """

    confidence = "native physical body/half pooling/E4M3 compact projection"

    def __init__(
        self,
        block_number: int,
        block: nn.Module,
        raw_values: torch.Tensor,
        ordinary_bytes: int,
    ) -> None:
        super().__init__()
        self.block_number = block_number
        self.block = block
        if isinstance(block, TorchPhysical1HBlock):
            self.confidence = (
                "SASS-static physical 1H body + pool->E4M3->32x64 transition"
            )
        if isinstance(block, TorchPhysical1HBlock):
            width = 32
        elif isinstance(block, TorchPhysical2HBlock):
            width = 64
            self.confidence = (
                "physical 2H attention + canonical FFN + pool->E4M3->64x128 transition"
            )
        elif isinstance(block, TorchPhysical4HBlock):
            width = 128
            self.confidence = (
                "physical 4H body + 2x2 pool/channel-route->128x256 transition"
            )
        elif isinstance(block, TorchPhysical8HBlock):
            width = 256
            self.confidence = (
                "physical 8H body + fused 2x2 pool/Q-route->256x512 transition"
            )
        else:
            raise TypeError(f"unsupported encoder body {type(block).__name__}")
        output_width = width * 2
        raw = raw_values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        transition_start = ordinary_bytes - 16
        transition = _decode_packed_qmma_projection(
            raw, transition_start, width, output_width
        )
        self.register_buffer("transition", torch.from_numpy(transition.copy()))
        if block_number == 4:
            self.block.quantize_output = False
            # Controlled compact signatures confirm ordinary projection N8
            # pairs are packed into physical 16-channel planes.
            n = np.arange(64, dtype=np.int64)
            output_route = (n//16)*16 + (n%2) + ((n//2)%2)*8 + ((n//4)%4)*2
            self.register_buffer("compact_output_route", torch.from_numpy(output_route))
            c_to_pixel = np.array([
                0,16,1,17,2,18,3,19,8,24,9,25,10,26,11,27,
                32,48,33,49,34,50,35,51,40,56,41,57,42,58,43,59,
                36,52,37,53,38,54,39,55,44,60,45,61,46,62,47,63,
                4,20,5,21,6,22,7,23,12,28,13,29,14,30,15,31], dtype=np.int64)
            self.register_buffer("pool_pixel_to_canonical", torch.from_numpy(np.argsort(c_to_pixel)))
            self.confidence = "controlled-SM89 physical 1H body + FP16 spatial pool/E4M3/QMMA + compact publication"

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.block_number == 4:
            full_half = self.block(feature)
            skip = quantize_e4m3(full_half)
            pixels = InputBlock._route_pixels(full_half, self.pool_pixel_to_canonical).to(torch.float16)
            top = pixels[:,0::2,0::2] + pixels[:,0::2,1::2]
            bottom = pixels[:,1::2,0::2] + pixels[:,1::2,1::2]
            pooled = quantize_e4m3(((top+bottom)*0.25).to(torch.float32))
            projected = quantize_e4m3((matmul_fp16_linear(pooled[..., self.block.channel_permutation], self.transition)).to(torch.float16))
            projected = projected[...,self.compact_output_route]
            batch,height,width,channels = projected.shape
            # The consumer's canonical 8x8 tiles each contain the next 64
            # row-major pooled pixels, not an ordinary spatial 8x8 window.
            compact = projected.reshape(batch,height//8,width//8,8,8,channels).permute(0,1,3,2,4,5).reshape(batch,height,width,channels)
            return compact,skip
        if ((self.block_number == 8 and isinstance(self.block, TorchPhysical2HBlock))
                or (self.block_number == 14 and isinstance(self.block, TorchPhysical4HBlock))
                or (self.block_number == 22 and isinstance(self.block, TorchPhysical8HBlock))):
            skip, transition_source = self.block.forward_with_prequant(feature)
        else:
            skip = self.block(feature)
            transition_source = skip
        if self.block_number == 8 and isinstance(self.block, TorchPhysical2HBlock):
            # The 2H DS kernel pools fused M bits 0/1, not adjacent elements
            # in the provisional canonical HWC view.  The remaining M bits
            # form the 4x4 compact tile consumed by the following 4H stage.
            batch, height, width, _ = skip.shape
            # SM89 DS 0x9ab0..0xa110 shuffles and pools the half
            # projection accumulators, not the separately published FP8 skip.
            internal = self.block._to_internal_tiles(transition_source).half().reshape(-1,16,4,64)
            pair0 = internal[:,:,0] + internal[:,:,1]
            pair1 = internal[:,:,2] + internal[:,:,3]
            pooled_internal = quantize_e4m3((pair0 + pair1) * 0.25)
            compact_internal = quantize_e4m3(matmul_fp16_linear(pooled_internal, self.transition))
            compact = (
                compact_internal.reshape(
                    batch, height // 8, width // 8, 4, 4, 128
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, height // 2, width // 2, 128)
            )
            return compact, skip
        if self.block_number == 22 and isinstance(self.block, TorchPhysical8HBlock):
            batch,height,width,_=skip.shape
            ph=(height+7)&-8;pw=(width+7)&-8;wh,ww=ph//8,pw//8
            if (ph,pw)!=(height,width):
                transition_source=self.block.expand_valid_tokens(transition_source)
            parts=[]
            for wy in range(wh):
                for wx in range(ww):
                    for ly in range(4):
                        for lx in range(4):
                            pair0 = (transition_source[:,wy+wh*(2*ly),wx+ww*(2*lx)].half()
                                     + transition_source[:,wy+wh*(2*ly),wx+ww*(2*lx+1)].half())
                            pair1 = (transition_source[:,wy+wh*(2*ly+1),wx+ww*(2*lx)].half()
                                     + transition_source[:,wy+wh*(2*ly+1),wx+ww*(2*lx+1)].half())
                            parts.append((pair0 + pair1) * 0.25)
            pooled=quantize_e4m3(torch.stack(parts,dim=1))
            compact=quantize_e4m3(
                matmul_fp16_linear(pooled.index_select(-1, self.block.ffn.input_route), self.transition)
            )
            compact=(compact.reshape(batch,wh,ww,4,4,512)
                     .permute(0,1,3,2,4,5).reshape(batch,ph//2,pw//2,512))
            return compact,skip
        if self.block_number == 14 and isinstance(self.block, TorchPhysical4HBlock):
            # DS 0x9380..0x9900 retains projection Half and adds vertical
            # pairs first. Exact (+64,-64,1/64,1/64) identity controls distinguish
            # this tree from horizontal/diagonal pairing before the /4 step.
            half = transition_source.half()
            pair0 = half[:,0::2,0::2] + half[:,1::2,0::2]
            pair1 = half[:,0::2,1::2] + half[:,1::2,1::2]
            pooled = quantize_e4m3((pair0 + pair1) * 0.25)
        else:
            pooled = quantize_e4m3(F.avg_pool2d(skip.permute(0,3,1,2),2).permute(0,2,3,1))
        if self.block_number == 14 and isinstance(self.block, TorchPhysical4HBlock):
            compact = quantize_e4m3(
                matmul_fp16_linear(pooled.index_select(-1, self.block.channel_route), self.transition)
            )
            # The compact producer writes four pooled values per 4H tile.
            # Its consumer is the native 8H fused view, not ordinary HWC.
            batch,ph,pw,channels=compact.shape
            compact=(compact.reshape(batch,ph//2,2,pw//2,2,channels)
                     .permute(0,1,3,2,4,5).reshape(batch,ph*pw//4,4,channels))
            # The consumer interleaves valid native slots, including the
            # short final half-band. Its spatial axes cross the producer's.
            ids=torch.from_numpy(TorchPhysical8HBlock.compact_plane_indices(pw,ph)).to(compact.device)
            compact=compact.permute(0,2,1,3).reshape(batch,ph*pw,channels)[:,ids].reshape(batch,pw,ph,channels)
            return compact, skip
        return matmul_fp16_linear(pooled, self.transition), skip


# Compatibility name for existing forensic scripts.
PackedBlock4Downsample = PackedEncoderDownsample


class DLSS5Encoder(nn.Module):
    """Real-weight executable encoder for blocks 0..22."""

    def __init__(self, root: Path = DEFAULT_ROOT, arena: Path = DEFAULT_ARENA) -> None:
        super().__init__()
        self.graph = RecoveredGraph(arena)
        archive = ExportedWeights(root)
        self.block0 = InputBlock(archive, arena)
        self.physical_1h = nn.ModuleDict({str(b):TorchPhysical1HBlock(root,b) for b in range(1,5)})
        self.physical_2h = nn.ModuleDict({str(b):TorchPhysical2HBlock(root,b) for b in range(5,9)})
        self.physical_4h = nn.ModuleDict({str(b):TorchPhysical4HBlock(b,archive.tensor(f"block{b}.layer0.layer")) for b in range(9,15)})
        self.physical_8h = nn.ModuleDict({str(b):TorchPhysical8HBlock(root,b) for b in range(15,23)})
        stages = [(self.physical_1h,4,20672),(self.physical_2h,8,61760),
                  (self.physical_4h,14,197184),(self.physical_8h,22,689232)]
        blocks = {}
        for family,terminal,ordinary_bytes in stages:
            for key,body in family.items():
                block = int(key)
                blocks[key] = (PackedEncoderDownsample(block,body,archive.tensor(f"block{block}.layer0.layer"),ordinary_bytes)
                               if block == terminal else body)
        self.blocks = nn.ModuleDict(blocks)

    @staticmethod
    def _trace(block: int, value: torch.Tensor, confidence: str) -> BlockTrace:
        finite = bool(torch.isfinite(value).all().item())
        safe = torch.nan_to_num(value.detach())
        return BlockTrace(
            block=block,
            shape=tuple(value.shape),
            finite=finite,
            absmax=float(safe.abs().max().item()),
            mean=float(safe.mean().item()),
            std=float(safe.std().item()),
            confidence=confidence,
        )

    def forward(self, inputs: torch.Tensor, *, collect_trace: bool = True) -> EncoderResult:
        trace: list[BlockTrace] = []
        skips: dict[int, torch.Tensor] = {}
        feature, skips[0] = self.block0(inputs)
        if collect_trace:
            trace.append(self._trace(0,feature,"SASS packet/FP16 adapter/physical pre-Swin and dual publication"))
        for block in range(1,23):
            module = self.blocks[str(block)]
            if block in (4,8,14,22):
                feature,skips[block] = module(feature)
            else:
                feature = module(feature)
                if block == 5:
                    feature = publish_first_2h(feature)
            if collect_trace:
                trace.append(self._trace(block,feature,module.confidence))
            if not torch.isfinite(feature).all():
                raise FloatingPointError(f"block{block} produced non-finite values")
        return EncoderResult(feature,skips,tuple(trace))


def format_trace(trace: Iterable[BlockTrace]) -> str:
    lines = []
    for item in trace:
        lines.append(
            f"block{item.block:02d} shape={str(item.shape):>20} "
            f"finite={str(item.finite):5s} absmax={item.absmax:.6g} "
            f"mean={item.mean:.6g} std={item.std:.6g} [{item.confidence}]"
        )
    return "\n".join(lines)


__all__ = [
    "Block70RuntimeInputs",
    "BlockTrace",
    "DLSS5Encoder",
    "EncoderResult",
    "ExportedWeights",
    "RecoveredGraph",
    "format_trace",
    "DLSS5Reconstruction",
    "MinimalModelResult",
    "ReconstructionResult",
    "PackedBlock4Downsample",
    "PackedEncoderDownsample",
    "quantize_e4m3",
]


# ---------------------------------------------------------------------------
# Full 71-block structural reconstruction
# ---------------------------------------------------------------------------

SPLIT_SHIFTED = {
    **{block: SWIN_PHASES[(block - 23) % 4] for block in range(23, 31)},
    **{block: SWIN_PHASES[(block - 40) % 4] for block in range(40, 48)},
}

DECODER_SHIFTED = {
    **{block: SWIN_PHASES[(block - 48) % 4] for block in range(48, 56)},
    # Runtime partitions: block56 X, then 57 Y / 58 none / 59 XY / 60 X / 61 Y.
    **{block: SWIN_PHASES[(block - 54) % 4] for block in range(56, 62)},
    **{block: SWIN_PHASES[(block - 62) % 4] for block in range(62, 66)},
    **{block: SWIN_PHASES[(block - 66) % 4] for block in range(66, 70)},
}


def _packed_f32(values: torch.Tensor, offset: int, count: int) -> torch.Tensor:
    half_count = count * 2
    array = np.frombuffer(
        values[offset : offset + half_count].cpu().numpy().astype("<f2", copy=False).tobytes(),
        dtype="<f4",
        count=count,
    ).copy()
    return torch.from_numpy(array)






def _bit_route_graph(size: int, destinations: tuple[int, ...]) -> np.ndarray:
    bits = (size - 1).bit_length()
    if len(destinations) != bits:
        raise ValueError((size, destinations))
    return np.array([
        sum(((value >> bit) & 1) << destinations[bit] for bit in range(bits))
        for value in range(size)
    ], dtype=np.int64)


def _decode_vit_qkv(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode the 0x80 scale prefix and K-plane/head/QKV-interleaved matrix."""
    if len(raw) != 0x300080:
        raise ValueError(f"ViT QKV record must be 0x300080 bytes, got {len(raw):#x}")
    byte = np.frombuffer(raw, dtype=np.uint8)
    scale = np.frombuffer(raw, dtype="<f4", count=32).astype(np.float32).copy()
    k = np.arange(1024, dtype=np.int64)[:, None]
    n = np.arange(1024, dtype=np.int64)[None, :]
    head, dim = n // 32, n % 32
    row, column = k % 32, dim % 16
    fragment = (
        16 * (4 * (column & 7) + ((row >> 2) & 3))
        + 8 * (column >> 3)
        + (row & 3)
        + 4 * (row >> 4)
    )
    base = (
        0x80 + (k // 32) * 0x18000 + head * 0xC00
        + (dim // 16) * 0x200 + fragment
    )
    matrices = np.stack([
        _decode_e4m3_numpy(byte[base + component * 0x400])
        for component in range(3)
    ])
    return scale, matrices


def _vit_activation(value: torch.Tensor) -> torch.Tensor:
    """MpCubicSilu: clamp only the gate input; multiply by the original z."""
    # Active cc_vit_1d_ffn_expand*_fp8 uses the same two HFMA2 gates
    # and final HMUL2 as fused Swin, not a full-FP32 cubic polynomial.
    return fast_activation_2h(value)


def _vit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   height: int, width: int) -> torch.Tensor:
    """ViT quantizes the exponent, accumulates P×V, THEN applies the reciprocal.

    Q rows stay in the model's logical view. K/V are gathered into the native
    fragment-token order, so the explicit K32 and denominator K64 boundaries
    refer to the right members. Repack99's E16 producer permutation cancels
    the attention load's E16 permutation; only lowXY swap/column-major remain.
    """
    tokens = height * width
    if height % 4 or width % 4 or tokens <= 0 or k.shape[-2] != tokens:
        raise ValueError((height, width, k.shape))
    fragment = torch.arange(tokens, device=k.device)
    x, y = fragment // height, fragment % height
    order = ((y & ~1) | (x & 1)) * width + ((x & ~1) | (y & 1))
    k = k.index_select(-2, order)
    v = v.index_select(-2, order)
    logits = q @ k.transpose(-1, -2)
    transformed = (
        logits.half().float() * 0.08953857421875 + 1.708984375
    ).half().clamp(1.439453125, 1.9775390625)
    bits = ((transformed.view(torch.int16).to(torch.int32) << 4) + 0x4000) & 0xFFFF
    exponent = bits.to(torch.int16).view(torch.float16)
    # Spatial padding is real QKV data. Only virtual keys beyond N, added by
    # the native K64 loop, are removed; their QK=0 exponent is exactly43/512.
    padded_tokens = (tokens + 63) // 64 * 64
    extra = padded_tokens - tokens
    denominator_input = F.pad(exponent, (0, extra), value=0.083984375) if extra else exponent
    total = torch.zeros_like(exponent[..., :1])
    for start in range(0, padded_tokens, 64):
        total = (total + sum_fp16_key64(denominator_input[..., start:start + 64], "vit")).half()
    if extra:
        total = (total - torch.full_like(total, extra * 0.083984375)).half()
    reciprocal = total.clamp_min(6.198883056640625e-05).float().reciprocal().half()
    # A final virtual K32 has V=0 and leaves a finite half accumulator intact.
    # Unlike denominator padding, that zero contribution can be omitted.
    numerator = matmul_fp16_accumulate(quantize_e4m3(exponent), v)
    return quantize_e4m3(numerator.half() * reciprocal)


class Vit1DWeights(nn.Module):
    """Runtime/SASS-decoded weights for one 1024-channel CCViT compound block."""

    def __init__(self, archive: ExportedWeights, block: int) -> None:
        super().__init__()
        layers = [archive.tensor(f"block{block}.layer{i}.layer") for i in range(5)]
        expected = (2097160, 2098176, 1572928, 1, 525312)
        if tuple(item.numel() for item in layers) != expected:
            raise ValueError(f"block{block}: unexpected CCVit1D record sizes")
        raw = [item.detach().cpu().numpy().astype("<f2", copy=False).tobytes() for item in layers]
        self.register_buffer(
            "ffn_expand", torch.from_numpy(_decode_packed_qmma_projection(raw[0], 0, 1024, 4096).copy())
        )
        self.register_buffer(
            "ffn_contract", torch.from_numpy(_decode_packed_qmma_projection(raw[1], 0, 4096, 1024).copy())
        )
        self.register_buffer(
            "ffn_skip", torch.from_numpy(np.frombuffer(raw[1], "<f2", 1024, 0x400000).astype(np.float32).copy())
        )
        scale, qkv = _decode_vit_qkv(raw[2])
        self.register_buffer("attn_scale", torch.from_numpy(scale))
        self.register_buffer("qkv_weight", torch.from_numpy(qkv.copy()))
        if layers[3].numel() != 1:
            raise ValueError(f"block{block}: unexpected attention marker size")
        self.register_buffer(
            "projection_weight", torch.from_numpy(_decode_packed_qmma_projection(raw[4], 0, 1024, 1024).copy())
        )
        self.register_buffer(
            "attn_skip", torch.from_numpy(np.frombuffer(raw[4], "<f2", 1024, 0x100000).astype(np.float32).copy())
        )
        route_i = _bit_route_graph(1024, (0, 1, 3, 4, 2, 5, 6, 7, 8, 9))
        route_c = _bit_route_graph(1024, (0, 3, 4, 1, 2, 5, 6, 7, 8, 9))
        inverse_i = np.argsort(route_i)
        self.register_buffer("contract_k_route", torch.from_numpy(_bit_route_graph(4096, (0, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11))))
        self.register_buffer("consumer_k_route", torch.from_numpy(_bit_route_graph(1024, (0, 3, 1, 2, 4, 5, 6, 7, 8, 9))))
        self.register_buffer("residual_raw_route", torch.from_numpy(inverse_i[route_c]))
        self.register_buffer("inverse_input_route", torch.from_numpy(inverse_i))
        self.register_buffer("next_input_route", torch.from_numpy(np.argsort(route_c)[route_i]))


class StructuralVit1D(nn.Module):
    confidence = (
        "runtime-closed 1024->4096->1024 FFN, full 32x32-head QKV, "
        "clamped bit-exponential global attention and 1024 projection"
    )

    def __init__(self, block: int, weights: Vit1DWeights) -> None:
        super().__init__()
        self.block = block
        self.weights = weights

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.shape[-1] != 1024:
            raise ValueError(f"block{self.block}: expected C=1024")
        batch, height, width, _ = feature.shape
        tokens = quantize_e4m3(feature).reshape(batch, height * width, 1024)
        block_input = tokens
        hidden = quantize_e4m3(_vit_activation(matmul_fp16_linear(tokens, self.weights.ffn_expand)))
        tokens = quantize_e4m3(matmul_fp16_splitk(
            hidden[..., self.weights.contract_k_route], self.weights.ffn_contract, 4,
            initial=block_input[..., self.weights.residual_raw_route].half() * self.weights.ffn_skip.half()
        ))
        qkv_input = tokens[..., self.weights.consumer_k_route]
        qkv = [matmul_fp16_splitk(qkv_input, matrix, 2) for matrix in self.weights.qkv_weight]
        q = qkv[0].reshape(batch, height * width, 32, 32).permute(0, 2, 1, 3)
        k = qkv[1].reshape(batch, height * width, 32, 32).permute(0, 2, 1, 3)
        v = qkv[2].reshape(batch, height * width, 32, 32).permute(0, 2, 1, 3)
        # cc_vit_1d_qkv_chained_fp8: half rsqrt normalization, then
        # HMUL2 by sqrt(32), then HMUL2 by the half learned head scale.
        q = quantize_e4m3(
            (normalize_fp16_2h(q) * 5.65625)
            * self.weights.attn_scale.reshape(1, 32, 1, 1).half()
        )
        k = quantize_e4m3(normalize_fp16_2h(k))
        v = quantize_e4m3(v)
        attended = _vit_attention(q, k, v, height, width)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, height * width, 1024)
        output = quantize_e4m3(matmul_fp16_splitk(
            attended[..., self.weights.consumer_k_route], self.weights.projection_weight, 4,
            initial=tokens.half() * self.weights.attn_skip.half()
        ))
        if self.block < 38:
            output = output[..., self.weights.next_input_route]
        return output.reshape(batch, height, width, 1024)


class UpsampleTransitionWeights(nn.Module):
    """Statically recovered 2K->K packed projection plus residual gate tail."""

    def __init__(self, prefix: torch.Tensor, width: int) -> None:
        super().__init__()
        matrix_half_count = width * width
        if prefix.numel() < matrix_half_count:
            raise ValueError(f"upsample C={width}: transition prefix too small")
        raw = prefix.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        matrix = _decode_packed_qmma_projection(raw, 0, width * 2, width)
        self.register_buffer("matrix", torch.from_numpy(matrix.copy()))
        auxiliary = prefix[matrix_half_count:]
        if auxiliary.numel() != width:
            raise ValueError(
                f"upsample C={width}: expected {width} FP16 tail values, "
                f"got {auxiliary.numel()}"
            )
        self.register_buffer("sin", auxiliary.contiguous())


def _encoder_to_decoder512_pixels(height:int,width:int,device:torch.device) -> torch.Tensor:
    """Source HWC pixel of each decoder fused512 token (exact address composition)."""
    if height%4 or width%4:raise ValueError((height,width))
    token=torch.arange(height*width,device=device,dtype=torch.long)
    band=token//(height*4);within=token%(height*4);slot=within%32
    y=(within//32)*8+((slot>>3)&1)+((slot&1)<<1)+(((slot>>4)&1)<<2)
    x=band*4+((slot>>1)&3)
    return y*width+x


class StructuralUpsample(nn.Module):
    confidence = (
        "SASS-proven integer 2x replication + packed 2K->K conv/dw projection "
        "+ named sin skip gate"
    )

    def __init__(
        self,
        block: int,
        width: int,
        transition: UpsampleTransitionWeights,
        swin: nn.Module | None,
    ) -> None:
        super().__init__()
        self.block = block
        self.width = width
        self.transition = transition
        self.swin = swin
        if block == 39:
            # SM89 identity-matrix signatures recover the K consumer route
            # generic bits -> main channel bits (0,3,1,2,4,5,6,7,8,9).
            self.register_buffer(
                "main_channel_route",
                torch.from_numpy(_bit_route_graph(1024, (0, 3, 1, 2, 4, 5, 6, 7, 8, 9))),
            )
            self.confidence = (
                "runtime-closed 2x replication/crop + packed 1024->512 QMMA "
                "+ skip sin gate; physical oracle corr 0.999977"
            )
        else:
            self.register_buffer("main_channel_route", torch.arange(width * 2))
        if block == 66:
            self.confidence = (
                "packed 64->32 projection + named sin skip gate recovered; "
                "tilesync scheduler object is backend runtime state, not serialized network data"
            )

    def forward(self, main: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[-1] != self.width:
            raise ValueError(f"block{self.block}: skip must have C={self.width}")
        if main.shape[-1] != self.width * 2:
            raise ValueError(
                f"block{self.block}: main must have C={self.width * 2}, got {main.shape[-1]}"
            )
        # CUBIN address generation uses integer source=(output//2), not a
        # ratio-based resize to the cropped destination extent.
        main = main.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        main = main[:, : skip.shape[1], : skip.shape[2], :]
        if main.shape[1:3] != skip.shape[1:3]:
            raise ValueError(
                f"block{self.block}: 2x replicated main {main.shape[1:3]} "
                f"cannot cover skip {skip.shape[1:3]}"
            )
        main = quantize_e4m3(main)
        source = main[..., self.main_channel_route]
        # Native block39 grid.Z partitions K into four contiguous256 ranges.
        # Each range starts at zero; the ordered tilesync chain adds its Half
        # partial to the preceding partial sum before the final skip HFMA.
        projected = (matmul_fp16_splitk(source, self.transition.matrix, 4)
                     if self.block == 39 else matmul_fp16_linear(source, self.transition.matrix))
        fused = quantize_e4m3(projected + skip * self.transition.sin)
        if self.block==39:
            b,h,w,c=fused.shape
            fused=fused.reshape(b,h*w,c)[:,_encoder_to_decoder512_pixels(h,w,fused.device)].reshape(b,h,w,c)
        return self.swin(fused) if self.swin is not None else fused


class TorchPhysicalUpsample48(nn.Module):
    """Physical 512->256 decoder transition plus its fused 8H body."""

    confidence = "controlled-runtime block48 transition and fused 8H body"
    numeric_confidence = confidence

    def __init__(self, root: Path, values: torch.Tensor) -> None:
        super().__init__()
        raw = values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        if len(raw) != 0xC8630:
            raise ValueError(len(raw))
        self.register_buffer("matrix", torch.from_numpy(
            _decode_packed_qmma_projection(raw, 0x58000, 512, 256).copy()
        ))
        self.register_buffer("sin", torch.from_numpy(
            np.frombuffer(raw, "<f2", count=256, offset=0x78200).astype(np.float32).copy()
        ))
        body = bytearray(0xA8450)
        body[0x00000:0x58000] = raw[0x00000:0x58000]
        body[0x58010:0x58210] = raw[0x78000:0x78200]
        body[0x58220:0xA8450] = raw[0x78400:0xC8630]
        self.body = TorchPhysical8HBlock(
            root, 48, phase=(0, 0),
            values=torch.from_numpy(np.frombuffer(body, "<f2").copy()),
        )
        p32 = np.asarray([
            0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
            16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
        ], dtype=np.int64)
        p512 = np.concatenate([p32 + 32 * group for group in range(16)])
        self.register_buffer("inverse_p512", torch.from_numpy(np.argsort(p512).copy()))

    @staticmethod
    def source_indices(height:int,width:int) -> np.ndarray:
        ph=(height+7)&-8;pw=(width+7)&-8;wh,ww=ph//8,pw//8
        index=TorchPhysical8HBlock.valid_token_indices(height,width)
        r,c=index//pw,index%pw;by,ly=r%wh,r//wh;bx,lx=c%ww,c//ww
        return 16*(bx*wh+by)+((ly>>2)&1)+(((lx>>1)&1)<<1)+(((lx>>2)&1)<<2)+(((ly>>1)&1)<<3)

    def forward(self,main:torch.Tensor,skip:torch.Tensor) -> torch.Tensor:
        b,h,w,c=skip.shape;mh,mw=main.shape[1:3]
        if c!=256 or main.shape[-1]!=512 or (mh,mw)!=(((h+7)//8)*4,((w+7)//8)*4):
            raise ValueError((main.shape,skip.shape))
        source=quantize_e4m3(main).reshape(b,mh*mw,512)
        projected=matmul_fp16_linear(source.index_select(-1, self.inverse_p512), self.matrix).half().float()
        ids=torch.from_numpy(self.source_indices(h,w)).to(main.device)
        transition=projected[:,ids].reshape(b,h,w,256)
        # Native projection stays half through the skip HFMA. The body then
        # stores fused E4M3 to shared memory, including its FFN residual path.
        return self.body(quantize_e4m3(transition+skip*self.sin))


class TorchPhysicalUpsample56(nn.Module):
    """Physical 256->128 decoder transition plus its fused 4H body."""

    confidence = "controlled-runtime block56 transition and fused 4H body"
    numeric_confidence = confidence

    def __init__(self, root: Path, values: torch.Tensor) -> None:
        super().__init__()
        raw = values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        if len(raw) != 0x38320:
            raise ValueError(len(raw))
        self.register_buffer("matrix", torch.from_numpy(
            _decode_packed_qmma_projection(raw, 0x18000, 256, 128).copy()
        ))
        self.register_buffer("sin", torch.from_numpy(
            np.frombuffer(raw, "<f2", count=128, offset=0x20100).astype(np.float32).copy()
        ))
        body = bytearray(0x30240)
        body[0x00000:0x18000] = raw[0x00000:0x18000]
        body[0x18010:0x18110] = raw[0x20000:0x20100]
        body[0x18120:0x30240] = raw[0x20200:0x38320]
        self.body = TorchPhysical4HBlock(
            56, torch.from_numpy(np.frombuffer(body, "<f2").copy())
        )
        self.body.layout_block = 11  # X phase, preceding block57's Y phase
        p32 = np.asarray([
            0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
            16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
        ], dtype=np.int64)
        p256 = np.concatenate([p32 + 32 * group for group in range(8)])
        self.register_buffer("inverse_p256", torch.from_numpy(np.argsort(p256).copy()))

    @staticmethod
    def source_indices(height:int,width:int) -> np.ndarray:
        if height%4 or width%4:raise ValueError((height,width))
        r,c=np.indices((height,width),dtype=np.int64)
        quarter=(((r//4)*(width//4)+c//4)*4+((r>>1)&1)*2+((c>>1)&1)).ravel()
        count=height*width//4
        raw_pixel=(quarter%4)*(count//4)+quarter//4
        # Invert the same quarter-plane publication used at block14.
        consumer_to_plane=TorchPhysical8HBlock.compact_plane_indices(width//2,height//2)
        return np.argsort(consumer_to_plane)[raw_pixel]

    def forward(self,main:torch.Tensor,skip:torch.Tensor) -> torch.Tensor:
        b,h,w,c=skip.shape;mh,mw=main.shape[1:3]
        if c!=128 or main.shape[-1]!=256 or (mh,mw)!=(w//2,h//2):
            raise ValueError((main.shape,skip.shape))
        source=quantize_e4m3(main).reshape(b,mh*mw,256)
        projected=matmul_fp16_linear(source.index_select(-1, self.inverse_p256), self.matrix).half().float()
        ids=torch.from_numpy(self.source_indices(h,w)).to(main.device)
        transition=projected[:,ids].reshape(b,h,w,128)
        return self.body(quantize_e4m3(transition+skip*self.sin))


class TorchPhysicalUpsample62(nn.Module):
    """Physical 128->64 decoder transition plus its fused 2H body."""

    confidence = "controlled-runtime block62 transition and fused 2H body"
    numeric_confidence = confidence

    def __init__(self, root: Path, values: torch.Tensor) -> None:
        super().__init__()
        raw = values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        if len(raw) != 0x111A0:
            raise ValueError(len(raw))
        self.register_buffer("matrix", torch.from_numpy(
            _decode_packed_qmma_projection(raw, 0x7000, 128, 64).copy()
        ))
        self.register_buffer("sin_physical", torch.from_numpy(
            np.frombuffer(raw, "<f2", count=64, offset=0x9080).astype(np.float32).copy()
        ))
        body = bytearray(0xF140)
        body[0x0000:0x7000] = raw[0x0000:0x7000]
        body[0x7010:0x7090] = raw[0x9000:0x9080]
        body[0x70A0:0xF0B0] = raw[0x9100:0x11110]
        body[0xF0B0:0xF130] = raw[0x11110:0x11190]
        self.body = TorchPhysicalDecoder2HBlock(
            root, 62, values=torch.from_numpy(np.frombuffer(body, "<f2").copy())
        )
        p32 = np.asarray([
            0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
            16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
        ], dtype=np.int64)
        p64 = np.r_[p32, p32 + 32]
        p128 = np.concatenate([p32 + 32 * group for group in range(4)])
        self.register_buffer("p64", torch.from_numpy(p64.copy()))
        self.register_buffer("inverse_p128", torch.from_numpy(np.argsort(p128).copy()))

    def forward(self, main: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = main.shape
        if channels != 128 or skip.shape[-1] != 64:
            raise ValueError((main.shape, skip.shape))
        if height%8 or width%8:raise ValueError((height,width))
        row, col = np.indices((height, width), dtype=np.int64)
        tile_y, tile_x = row // 4, col // 4
        local_y, local_x = row % 4, col % 4
        y=(tile_y//2)+(height//8)*((local_x>>1)+2*(local_y>>1))
        low = (
            (y & 1) | ((local_y & 1) << 1)
            | ((tile_x & 1) << 2) | ((local_x & 1) << 3)
        )
        tile_y_outer = (tile_y & 1) + 2 * (y >> 1)
        storage = low + 16 * ((tile_x >> 1) + (width // 8) * tile_y_outer)
        storage_index = torch.from_numpy(storage.reshape(-1)).to(device=main.device)
        internal = torch.empty(
            (batch, height * width, 128), dtype=main.dtype, device=main.device
        )
        internal[:, storage_index] = quantize_e4m3(main).reshape(batch, -1, 128)
        projected = matmul_fp16_linear(internal.index_select(-1, self.inverse_p128), self.matrix).half().float()
        out_height, out_width = height * 2, width * 2
        row, col, channel = np.indices((out_height, out_width, 64), dtype=np.int64)
        y=(row//16)+(out_height//16)*(channel>>4)
        low = (
            (y & 1) | ((row & 1) << 1)
            | (((col >> 3) & 1) << 2) | (((channel >> 2) & 1) << 3)
        )
        major=((row>>3)&1)+2*(y>>1)
        source_token=low+16*(col//16+(out_width//16)*major)
        source_channel = (
            (channel & 1) | (((col >> 1) & 1) << 1)
            | (((channel >> 1) & 1) << 2) | ((col & 1) << 3)
            | (((channel >> 3) & 1) << 4) | (((row >> 2) & 1) << 5)
        )
        source_token_t = torch.from_numpy(source_token.reshape(-1)).to(device=main.device)
        source_channel_t = torch.from_numpy(source_channel.reshape(-1)).to(device=main.device)
        projected_channel = self.p64.index_select(0, source_channel_t)
        transition = projected[:, source_token_t, projected_channel].reshape(
            batch, out_height, out_width, 64
        )
        gate = (
            (channel & 1) | ((col & 1) << 1) | (((col >> 1) & 1) << 2)
            | (((channel >> 1) & 1) << 3) | (((channel >> 3) & 1) << 4)
            | (((row >> 2) & 1) << 5)
        )
        gate_t = torch.from_numpy(gate.reshape(-1)).to(device=main.device)
        sin = self.sin_physical.index_select(0, gate_t).reshape(
            out_height, out_width, 64
        )
        if skip.shape[1:3] != (out_height, out_width):
            raise ValueError((main.shape, skip.shape))
        return self.body(quantize_e4m3(transition + skip * sin))


class TorchPhysicalUpsample66(nn.Module):
    """Physical 64->32 decoder transition plus its fused 1H body.

    Controlled transition identities recover the source token/N formulas and
    the packed-matrix K route; five skip-gate probes recover the spatially
    invariant physical gate route.  The runtime record interleaves transition
    and body fields, so it is reconstructed explicitly instead of sliced as a
    prefix followed by an ordinary body.
    """

    confidence = "controlled-runtime block66 transition and fused 1H body"
    numeric_confidence = confidence

    def __init__(self, root: Path, values: torch.Tensor) -> None:
        super().__init__()
        raw = values.detach().cpu().numpy().astype("<f2", copy=False).tobytes()
        if len(raw) != 0x5900:
            raise ValueError(len(raw))
        matrix = _decode_packed_qmma_projection(raw, 0x2000, 64, 32)
        self.register_buffer("matrix", torch.from_numpy(matrix.copy()))
        sin_physical = np.frombuffer(raw, "<f2", count=32, offset=0x2860).astype(np.float32)
        sin_route = np.asarray([
            0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,
            4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31,
        ], dtype=np.int64)
        self.register_buffer("sin", torch.from_numpy(sin_physical[sin_route].copy()))
        body = bytearray(0x50C0)
        body[0x0000:0x2000] = raw[0x0000:0x2000]
        body[0x2010:0x2050] = raw[0x2810:0x2850]
        body[0x2060:0x5070] = raw[0x28A0:0x58B0]
        body[0x5070:0x50B0] = raw[0x58B0:0x58F0]
        body_values = torch.from_numpy(np.frombuffer(body, "<f2").copy())
        self.body = TorchPhysical1HBlock(root, 66, values=body_values)
        p32 = np.asarray([
            0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
            16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31,
        ], dtype=np.int64)
        self.register_buffer("p32", torch.from_numpy(p32.copy()))
        self.register_buffer("inverse_p64", torch.from_numpy(np.argsort(np.r_[p32,p32+32]).copy()))

    def forward(self, main: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = main.shape
        if channels != 64 or skip.shape[-1] != 32:
            raise ValueError((main.shape, skip.shape))
        if height%16 or width%16:raise ValueError((height,width))
        token_np, output_n_np = TorchPhysicalDecoder2HBlock._indices(height, width)
        token = torch.from_numpy(token_np).to(device=main.device)
        output_n = torch.from_numpy(output_n_np).to(device=main.device)
        source = quantize_e4m3(main).reshape(batch, -1)
        internal_n = torch.empty(
            (batch, height * width, 64), dtype=source.dtype, device=source.device
        )
        internal_n[:, token, output_n] = source
        projected = matmul_fp16_linear(internal_n.index_select(-1, self.inverse_p64), self.matrix).half().float()
        out_height, out_width = height * 2, width * 2
        row, col, channel = np.indices((out_height, out_width, 32), dtype=np.int64)
        low = (
            (((row >> 1) ^ (row >> 2)) & 1)
            | (((col >> 2) & 1) << 1)
            | (((row >> 2) & 1) << 2)
            | ((col & 1) << 3)
            | (((col >> 3) & 1) << 4)
            | (((col >> 4) & 1) << 5)
        )
        source_token = low + 64 * (
            col // 32 + (out_width // 32) * (row // 8)
        )
        source_n = (
            (channel & 1)
            | (((channel >> 4) & 1) << 1)
            | (((channel >> 1) & 1) << 2)
            | (((channel >> 3) & 1) << 3)
            | (((channel >> 2) & 1) << 4)
        )
        gather_token = torch.from_numpy(source_token.reshape(-1)).to(device=main.device)
        gather_n = self.p32.index_select(
            0, torch.from_numpy(source_n.reshape(-1)).to(device=main.device)
        )
        transition = projected[:, gather_token, gather_n].reshape(
            batch, out_height, out_width, 32
        )
        if skip.shape[1:3] != (out_height, out_width):
            raise ValueError((main.shape, skip.shape))
        # The mixed1H kernel retains the HFMA fusion result for its W2
        # residual. W1 alone crosses E4M3 at entry to the fused body.
        fused = (transition + skip.half().float() * self.sin.half().float()).half().float()
        return self.body(fused)


class StructuralPostBlock(nn.Module):
    confidence = (
        "block70 SASS-static physical 1H core, four-channel FP16 head, "
        "Color/MVec/PrevOutput five-tap history compositor and blend order recovered"
    )

    def __init__(self, archive: ExportedWeights) -> None:
        super().__init__()
        values = archive.tensor("block70.layer0.layer")
        if values.numel() != 10904:
            raise ValueError("block70: unexpected layer size")
        # The post record is a physical fused layout. FFN begins at byte zero;
        # two input gates inserted after its residual gate fuse block69 main with
        # encoder block0 skip. The ordinary QKV/bias/projection family is shifted
        # by +0x70 and is decoded by the block70 physical 1H specialization.
        post_root = archive.directory.parent
        raw_numpy = values.detach().cpu().numpy()
        main_gate_physical = np.frombuffer(
            raw_numpy.astype("<f2", copy=False).tobytes(), dtype="<f2", count=32, offset=0x2050
        )
        skip_gate_physical = np.frombuffer(
            raw_numpy.astype("<f2", copy=False).tobytes(), dtype="<f2", count=32, offset=0x2090
        )
        self.swin = TorchPhysical1HBlock(root=post_root, block=70, quantize_output=False)
        canonical_to_n = self.swin.output_inverse.cpu().numpy()[self.swin.channel_inverse.cpu().numpy()]
        # Main/skip gates are stored in projection-N order. Two controlled
        # 32-level one-hot signatures independently establish both vectors.
        self.register_buffer("input_scale", torch.from_numpy(main_gate_physical[canonical_to_n].astype(np.float32)))
        self.register_buffer("input_sin", torch.from_numpy(skip_gate_physical[canonical_to_n].astype(np.float32)))
        c_to_pixel = np.array([
            0,16,1,17,2,18,3,19,8,24,9,25,10,26,11,27,
            32,48,33,49,34,50,35,51,40,56,41,57,42,58,43,59,
            36,52,37,53,38,54,39,55,44,60,45,61,46,62,47,63,
            4,20,5,21,6,22,7,23,12,28,13,29,14,30,15,31],dtype=np.int64)
        self.register_buffer("canonical_to_pixel",torch.from_numpy(c_to_pixel))
        self.register_buffer("pixel_to_canonical",torch.from_numpy(np.argsort(c_to_pixel)))


        # SM89 89d0/8a70 load B halves for two K16 HMMA instructions.
        # A follows the projection's output-N order, not the QKV-K order.
        # Each lane stores K=(lane%4)*2+slot%2+8*(slot//2), N=lane//4.
        readout_parts = []
        raw_bytes = raw_numpy.astype('<f2', copy=False).tobytes()
        for offset in (0x5130, 0x5330):
            lanes = np.frombuffer(raw_bytes, dtype='<f2', count=256, offset=offset).reshape(32,8)[:,:4]
            matrix = np.empty((16,8),dtype=np.float32)
            for lane in range(32):
                for slot in range(4):
                    matrix[(lane%4)*2+(slot%2)+8*(slot//2),lane//4]=lanes[lane,slot]
            if np.any(matrix[:,4:] != 0):
                raise ValueError('block70: unexpected nonzero unused HMMA output columns')
            readout_parts.append(matrix[:,:4])
        readout_n = np.concatenate(readout_parts,axis=0)
        canonical_to_n = self.swin.output_inverse.cpu().numpy()[self.swin.channel_inverse.cpu().numpy()]
        self.register_buffer('head_n_to_canonical',torch.from_numpy(np.argsort(canonical_to_n)))
        self.register_buffer("out_conv_weight", torch.from_numpy(readout_n[canonical_to_n].T.copy()))
        blend = archive.tensor("block70.layer0.blend_scale")
        self.register_buffer("blend_scale", blend.reshape(()))

    def forward_head(
        self,
        main: torch.Tensor,
        enc0_skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the learned block70 core and return its complete four-channel head."""
        if main.shape[-1] != 32 or enc0_skip.shape[-1] != 32:
            raise ValueError(f"block70 expects C=32 inputs: {main.shape} vs {enc0_skip.shape}")
        if main.shape[:1] != enc0_skip.shape[:1]:
            raise ValueError(f"block70 batch mismatch: {main.shape} vs {enc0_skip.shape}")
        if main.shape[1:3] != enc0_skip.shape[1:3]:
            expected = (main.shape[1] * 2, main.shape[2] * 2)
            if tuple(enc0_skip.shape[1:3]) != expected:
                raise ValueError(f"block70 spatial inputs disagree: {main.shape} vs {enc0_skip.shape}")
            # The post kernel launches at full resolution and consumes the
            # half-resolution block69 chain together with block0's full skip.
            # The native post input consumes block69's inverse-first-1H
            # publication. In canonical coordinates its action is nearest
            # replication in real pixels, not nearest in the TinLayout view.
            real_main = InputBlock._route_pixels(main,self.pixel_to_canonical)
            real_main = real_main.repeat_interleave(2,1).repeat_interleave(2,2)
            main = InputBlock._route_pixels(real_main,self.canonical_to_pixel)
        # Both native post variants round the main product with HMUL2,
        # then HFMA2(skip, skip_gate, main_product). Do not round skip*gate
        # separately before adding it (all32 packed input paths are traced).
        main_product = (main.half() * self.input_scale.half()).float()
        fused = (enc0_skip.half().float() * self.input_sin.half().float()
                 + main_product).half().float()
        latent = self.swin(fused)
        source_n = latent[...,self.head_n_to_canonical]
        weight_n = self.out_conv_weight[:,self.head_n_to_canonical].T
        first = (source_n[...,:16] @ weight_n[:16]).to(torch.float16)
        head = (first.float() + source_n[...,16:] @ weight_n[16:]).to(torch.float16).float()
        # Three controlled RGB address rounds establish this complete native
        # head -> visible-pixel map; all262144 observed positions match exactly.
        head = InputBlock._route_pixels(head,self.pixel_to_canonical)
        return head, latent

    def forward(
        self,
        main: torch.Tensor,
        enc0_skip: torch.Tensor,
        runtime_inputs: Block70RuntimeInputs | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        head, latent = self.forward_head(main, enc0_skip)
        residual_rgb = head[..., :3]
        history_logit = head[..., 3:4]
        if runtime_inputs is None:
            # Diagnostic output only: this is the real network head, not a guessed
            # color blend.  Final RGB requires caller Color/PrevOutput/MVec.
            output = residual_rgb
            runtime_complete = False
        else:
            output = reconstruct_block70_color(
                residual_rgb,
                history_logit,
                runtime_inputs,
                self.blend_scale,
            )
            runtime_complete = True
        return output, latent, runtime_complete, head


@dataclass(frozen=True)
class MinimalModelResult:
    """The fixed Preset #1 learned model, excluding host pre/post mapping."""

    head: torch.Tensor
    latent: torch.Tensor
    trace: tuple[BlockTrace, ...]


@dataclass(frozen=True)
class ReconstructionResult:
    output: torch.Tensor
    latent: torch.Tensor
    trace: tuple[BlockTrace, ...]
    block70_runtime_complete: bool
    head: torch.Tensor


class DLSS5Reconstruction(nn.Module):
    """Full block0..70 pure-Torch reconstruction from local static evidence.

    Canonical Torch arithmetic replaces CUBIN instruction-level rounding.  The
    network graph, learned operators, visible quantization boundaries, frame
    input construction and final temporal compositor are all represented here;
    an independent NVIDIA numeric oracle is not available for error measurement.
    """

    def __init__(self, root: Path = DEFAULT_ROOT, arena: Path = DEFAULT_ARENA) -> None:
        super().__init__()
        self.encoder = DLSS5Encoder(root, arena)
        archive = ExportedWeights(root)
        split_modules: dict[str, nn.Module] = {}
        for block in range(23,31):
            split_modules[str(block)]=TorchPhysicalSplitSwin16H(root,block)
        for block in range(40,48):
            split_modules[str(block)]=TorchPhysicalSplitSwin16H(root,block)
        self.split=nn.ModuleDict(split_modules)
        self.vit = nn.ModuleDict({
            str(block): StructuralVit1D(block, Vit1DWeights(archive, block))
            for block in range(31, 39)
        })
        block39 = archive.tensor("block39.layer0.layer")
        transition39 = UpsampleTransitionWeights(block39, 512)
        self.up39 = StructuralUpsample(39, 512, transition39, None)

        decoder_widths = {
            **{block: 256 for block in range(48, 56)},
            **{block: 128 for block in range(56, 62)},
            **{block: 64 for block in range(62, 66)},
            **{block: 32 for block in range(66, 70)},
        }
        up_blocks = {48, 56, 62, 66}
        decoder: dict[str, nn.Module] = {}
        for block, width in decoder_widths.items():
            values = archive.tensor(f"block{block}.layer0.layer")
            if block in up_blocks:
                factory={48:TorchPhysicalUpsample48,56:TorchPhysicalUpsample56,62:TorchPhysicalUpsample62,66:TorchPhysicalUpsample66}[block]
                decoder[str(block)]=factory(root,values)
            elif 49 <= block <= 55:
                decoder[str(block)] = TorchPhysical8HBlock(
                    root, block, phase=DECODER_SHIFTED[block]
                )
            elif 57 <= block <= 61:
                decoder[str(block)] = TorchPhysical4HBlock(block, values)
            elif 63 <= block <= 65:
                decoder[str(block)] = TorchPhysicalDecoder2HBlock(root, block, values=values)
            elif 67 <= block <= 69:
                decoder[str(block)] = TorchPhysical1HBlock(root, block, values=values)
            else:
                raise ValueError(f"unsupported decoder block{block}")
        self.decoder = nn.ModuleDict(decoder)
        self.post = StructuralPostBlock(archive)

    @staticmethod
    def _append_trace(
        trace: list[BlockTrace], block: int, value: torch.Tensor, confidence: str
    ) -> None:
        trace.append(DLSS5Encoder._trace(block, value, confidence))
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"block{block} produced non-finite values")

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        *,
        collect_trace: bool = True,
        block70_runtime: Block70RuntimeInputs | None = None,
    ) -> ReconstructionResult:
        if inputs is None:
            if block70_runtime is None:
                raise ValueError("provide either a prepared [B,16,H,W] packet or block70_runtime resources")
            inputs = build_preblock_features(block70_runtime)
        encoded = self.encoder(inputs, collect_trace=collect_trace)
        trace = list(encoded.trace)
        feature = encoded.main
        skips = dict(encoded.skips)

        for block in range(23, 31):
            module = self.split[str(block)]
            # The terminal proj_pool consumes its projection Half directly;
            # its separate FP8 skip is formed in terminal_downsample.
            feature=module(feature,quantize_output=block != 30)
            if collect_trace:
                self._append_trace(trace,block,feature,module.confidence)
        block30 = self.split["30"]
        assert isinstance(block30,TorchPhysicalSplitSwin16H)
        feature,skips[30]=block30.terminal_downsample(feature)
        # Replace block30 trace with descriptor output0, not its pre-pool output1.
        if collect_trace:
            trace[-1]=DLSS5Encoder._trace(30,feature,block30.confidence)

        feature=feature.index_select(-1,self.vit['31'].weights.next_input_route)
        for block in range(31, 39):
            module = self.vit[str(block)]
            feature = module(feature)
            if collect_trace:
                self._append_trace(trace, block, feature, module.confidence)

        feature = self.up39(feature, skips[30])
        if collect_trace:
            self._append_trace(trace, 39, feature, self.up39.confidence)
        for block in range(40, 48):
            module = self.split[str(block)]
            feature = module(feature)
            if collect_trace:
                self._append_trace(trace, block, feature, module.confidence)

        skip_for_up = {48: 22, 56: 14, 62: 8, 66: 4}
        for block in range(48, 70):
            module = self.decoder[str(block)]
            if block in skip_for_up:
                assert isinstance(module, (StructuralUpsample, TorchPhysicalUpsample48, TorchPhysicalUpsample56, TorchPhysicalUpsample62, TorchPhysicalUpsample66))
                feature = module(feature, skips[skip_for_up[block]])
                confidence = module.confidence
            else:
                feature = module(feature)
                confidence = module.numeric_confidence
            if collect_trace:
                self._append_trace(trace, block, feature, confidence)

        output, latent, runtime_complete, head = self.post(
            feature, skips[0], block70_runtime
        )
        if collect_trace:
            confidence = self.post.confidence + (
                "; caller runtime color inputs bound"
                if runtime_complete else
                "; complete learned head4 (host pre/post resources not bound)"
            )
            self._append_trace(trace, 70, output if runtime_complete else head, confidence)
        return ReconstructionResult(output, latent, tuple(trace), runtime_complete, head)

    def forward_minimal(
        self,
        prepared_features: torch.Tensor,
        *,
        collect_trace: bool = False,
    ) -> MinimalModelResult:
        """Execute only the fixed Preset #1 learned model contract.

        ``prepared_features`` is the model's sixteen-channel pre-HMMA BCHW packet.
        Resource mapping, history ownership, UI/HDR handling and final color composition
        deliberately remain outside this minimal interface. The returned BHWC
        head contains three learned RGB residuals and one history logit.
        """
        result = self(
            prepared_features,
            collect_trace=collect_trace,
            block70_runtime=None,
        )
        return MinimalModelResult(result.head, result.latent, result.trace)




FrameInputs = Block70RuntimeInputs


class DLSS5Model(DLSS5Reconstruction):
    """Public single-file model loaded directly from ``weights_ht_blob.bin``.

    The binary is parsed and decoded automatically. The temporary decoded cache
    belongs to this model instance and is removed when the instance is released.
    """

    def __init__(self, weights_bin: str | Path) -> None:
        source = Path(weights_bin).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        temporary = tempfile.TemporaryDirectory(prefix="dlss5-torch-", ignore_cleanup_errors=True)
        root = Path(temporary.name)
        try:
            _materialize_original_weights(source, root)
            super().__init__(root=root, arena=root)
        except Exception:
            temporary.cleanup()
            raise
        self._weights_bin = source
        self._decoded_cache = temporary

    @property
    def weights_path(self) -> Path:
        return self._weights_bin

    @torch.inference_mode()
    def infer_minimal(
        self,
        prepared_features: torch.Tensor,
        *,
        collect_trace: bool = False,
        return_result: bool = False,
    ) -> torch.Tensor | MinimalModelResult:
        """Run the minimal learned model: BCHW sixteen-channel packet -> BHWC head4."""
        device = next(self.buffers()).device
        result = self.forward_minimal(
            prepared_features.to(device),
            collect_trace=collect_trace,
        )
        return result if return_result else result.head

    @torch.inference_mode()
    def infer_frame(
        self,
        frame: FrameInputs,
        *,
        collect_trace: bool = False,
        return_result: bool = False,
    ) -> torch.Tensor | ReconstructionResult:
        """Run Color/PrevOutput/MVec through block0..70 and return RGB output."""
        device = next(self.buffers()).device
        moved: dict[str, object] = {}
        for field in fields(frame):
            value = getattr(frame, field.name)
            moved[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        runtime = FrameInputs(**moved)
        result = self(None, collect_trace=collect_trace, block70_runtime=runtime)
        return result if return_result else result.output


def load_model(
    weights_bin: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> DLSS5Model:
    """Load the original exported BIN and construct the complete Torch model."""
    model = DLSS5Model(weights_bin).to(device)
    return model.eval() if eval_mode else model


__all__ = [
    "BlockTrace", "DLSS5Model", "FrameInputs", "MinimalModelResult",
    "ReconstructionResult", "format_trace", "load_model", "pad_color_for_neural_buffer",
]
