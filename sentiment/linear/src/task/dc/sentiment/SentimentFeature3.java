package task.dc.sentiment;

import java.util.*;

import task.dc.*;
import text.*;
import training.*;

public class SentimentFeature3 extends FeatureCounter{
	// Unigram + Bigram
	public HashMap<Integer,Integer> get_one(Paragraph p,Dict d){
		// --- not good design of public sents and words, but ...
		HashMap<Integer,Integer> feats = new HashMap<Integer,Integer>();
		for(Sentence s : p.sents){
			//first deal with negation
			negation(s);
			for(int i=0;i<s.words.size();i++){
				//unigram
				String str = s.words.get(i);
				add_one_str_feature(str,d,feats);
				//bigram
				if(i < s.words.size()-1){
					String str2 = str+"||"+s.words.get(i+1);
					add_one_str_feature(str2,d,feats);
				}
			}
		}
		return feats;
	}
}
