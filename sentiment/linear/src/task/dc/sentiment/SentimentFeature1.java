package task.dc.sentiment;

import java.util.*;

import task.dc.*;
import text.*;
import training.*;

public class SentimentFeature1 extends FeatureCounter{
	//baseline feature --- bag of words
	public HashMap<Integer,Integer> get_one(Paragraph p,Dict d){
		//bag of words 
		// --- not good design of public sents and words, but ...
		HashMap<Integer,Integer> feats = new HashMap<Integer,Integer>();
		for(Sentence s : p.sents){
			//first deal with negation
			negation(s);
			for(String str : s.words){
				add_one_str_feature(str,d,feats);
			}
		}
		return feats;
	}
}
