echo "running all.sh..."
sh PreDecision/self.sh
CC=g++
$CC -o PreDecision/friend PreDecision/friend.cpp
./PreDecision/friend
rm -rf PreDecision/friend
sh Tweets/target_user/self.sh
sh UserGraph/target_user.sh
sh Tweets/target_user/friend.sh
sh Tweets/big_user.sh
echo "This is the end of all"