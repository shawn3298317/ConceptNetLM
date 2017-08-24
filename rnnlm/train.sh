#!/bin/bash

#declare -a users=("_supernatural_" "aclaysuper" "cbreezy_4ever" "celebfanspage" "crazybsbfan31" "daddony58" "demi_loveatox3" "drubisunirun" "forensicmama" "gmorataya" "hannlawrence" "heather_wolf" "iamnappy_901xxx" "jaxsk" "kush_420_report" "lalalovato" "mainlabswebsite" "marytorres12" "milessellydemz" "milla_swe" "missoxygenrina3" "mitchel_emily4e" "mjjalways" "mystyle" "nayrod" "officialas" "polascheps" "princessgwenie" "protruckr" "rj_acosta" "rpattzpicspam" "sallytheshizzle" "sandrawelling" "seemasugandh" "sexmenickj" "shoescom_sales" "slworking" "surinotes" "swtgeorgiabrwn" "teampadalecki" "tweetypie54" "youuarenotalone")
export PATH=/usr/local/cuda/bin:${PATH}

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
do
	echo "Train user $i: without feature"
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_user_feat.py -user $i
	sleep 10
done

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
do
	echo "Train user $i: with pos feature"
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_user_feat.py -feat pos -user $i
	sleep 10
done

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
do
	echo "Train user $i: withsem feature"
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_user_feat.py -feat sem -user $i
	sleep 10
done