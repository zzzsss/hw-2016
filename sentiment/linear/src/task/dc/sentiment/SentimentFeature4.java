package task.dc.sentiment;

import java.util.*;

import task.dc.*;
import text.*;
import training.*;

public class SentimentFeature4 extends FeatureCounter{
	//adding the although-but dealing...
	public HashMap<Integer,Integer> get_one(Paragraph p,Dict d){
		// unigram + bigram + although-but
		HashMap<Integer,Integer> feats = new HashMap<Integer,Integer>();
		for(Sentence s : p.sents){
			negation(s);
			
			int size = s.words.size();
			for(int i=0;i<size;i++){
				String c_word = s.words.get(i);
				//unigram
				add_one_str_feature(c_word,d,feats);
				//bigram
				if(i < s.words.size()-1){
					String str2 = c_word+"||"+s.words.get(i+1);
					add_one_str_feature(str2,d,feats);
				}
			}
			
			deal_but(s);
			for(int i=0;i<size;i++){
				String c_word = s.words.get(i);
				//unigram
				add_one_str_feature(c_word,d,feats);
				//bigram
				if(i < s.words.size()-1){
					String str2 = c_word+"||"+s.words.get(i+1);
					add_one_str_feature(str2,d,feats);
				}
			}
			
		}
		return feats;
	}
}
