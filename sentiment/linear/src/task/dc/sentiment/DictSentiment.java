package task.dc.sentiment;

import java.io.FileInputStream;
import java.util.*;

import text.Paragraph;
//the sentiment dictionary
public class DictSentiment {
	public static int BAD_SET=0,GOOD_SET=1,ALL_SET=2;
	public static DictSentiment dict_sentiment;
	static{
		String good_file = "lib/senti_dict/pos.txt";
		String bad_file = "lib/senti_dict/neg.txt";
		dict_sentiment = new DictSentiment();
		dict_sentiment.read(good_file, bad_file);
	}
	
	
	HashSet<String> good_words;
	HashSet<String> bad_words;
	public DictSentiment(){
		good_words = new HashSet<String>();
		bad_words = new HashSet<String>();
	}
	
	public void read(String good_file,String bad_file){
		read(good_file,true);
		read(bad_file,false);
	}
	public void read(String file,boolean good){
		try{
			FileInputStream f = new FileInputStream(file);
			Scanner s = new Scanner(f);
			while(s.hasNext()){
				String x = s.next();
				add(x,good);
			}
			s.close();
		}
		catch (Exception e){
            e.printStackTrace();
		}
	}
	public void add(String w,boolean good){
		if(good)
			good_words.add(w);
		else
			bad_words.add(w);
	}
	
	public int search(String w){
		if(good_words.contains(w))
			return GOOD_SET;
		else if(bad_words.contains(w))
			return BAD_SET;
		else
			return -1;
	}
	public List<String> get_bunch(int which){
		if(which == BAD_SET)
			return new ArrayList<String>(good_words);
		else if(which == GOOD_SET)
			return new ArrayList<String>(bad_words);
		else if(which == ALL_SET){
			ArrayList<String> t = new ArrayList<String>();
			t.addAll(good_words);
			t.addAll(bad_words);
			return t;
		}
		else
			return null;
	}
}
