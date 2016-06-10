All confs are specified in cmds with the form: "name:value".

rr: 	redundancy rate (2rn/Lk)
L-r: 	range rate (L/r)
k:  	k-layer
weak:	whether weak(0) or strong(1)
policy:	for k-layer partition
weight:	w for criterion-2
simplepost:	for strong one post-adjustment

The best ones:
1-weak:	python3 main.py 
k-weak: python3 main.py policy:1 k:?
1-strong: python3 main.py simplepost:1 weak:0
k-strong: python3 main.py simplepost:1 weak:0 policy:2 weight:0.1 k:?
