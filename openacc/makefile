q2b_seq: q2b.cpp helper.cpp
	git pull
	g++ q2b.cpp -o q2b_seq

q3a_seq: q3a.cpp helper.cpp
	git pull
	g++ q3a.cpp -o q3a_seq

q3b_seq: q3b.cpp helper.cpp
	git pull
	g++ q3b.cpp -o q3b_seq

q2b_para: q2b.cpp helper.cpp
	git pull
	pgc++ -acc q2b.cpp -Minfo=accel -o q2b_para

q3a_para: q3a.cpp helper.cpp
	git pull
	pgc++ -acc q3a.cpp -Minfo=accel -o q3a_para

q3b_para: q3b.cpp helper.cpp
	git pull
	pgc++ -acc q3a.cpp -Minfo=accel -o q3b_para

all_seq: q3b_seq q3a_seq q2b_seq
	git pull
	make q2b_seq
	make q3a_seq
	make q3b_seq

all_parallel: q3b_seq q3a_seq q2b_seq
	git pull
	make q2b_para
	make q3a_para
	make q3b_para

clean:
	rm -f q2b_seq
	rm -f q2b_para
	rm -f q3a_seq
	rm -f q3a_para
	rm -f q3b_seq
	rm -f q3b_para
