package task.dc.sentiment;

import java.util.*;

import task.dc.*;
import text.*;
import training.*;

public class SentimentFeature_try extends FeatureCounter{
	//baseline feature --- bag of words
	public HashMap<Integer,Integer> get_one(Paragraph p,Dict d){
		//bag of words 
		// --- not good design of public sents and words, but ...
		HashMap<Integer,Integer> feats = new HashMap<Integer,Integer>();
		for(Sentence s : p.sents){
			/*
			boolean contains = false;
			//whether has the word
			for(String w : s.words){
				if(DictSentiment.dict_sentiment.search(w)>0){
					contains = true;
					break;
				}
			}
			if(! contains)
				continue;
			*/
			
			//first deal with negation
			negation(s);
			
			int size = s.words.size();
			for(int i=0;i<size;i++){
				String c_pos = s.pos.get(i);
				String c_word = s.words.get(i);
				
				//position --- 3 places
				/*
				if(c_pos.length() > 0)
					if(c_pos.charAt(0)=='n'||c_pos.charAt(0)=='v'||c_pos.charAt(0)=='a'||c_pos.charAt(0)=='d'){
						int position = (int)((i*3.0)/size);
						if(position == 2)
							add_one_str_feature(c_word+"_POSI_"+position,d,feats);
					}
				*/
				
				
				//unigram
				add_one_str_feature(c_word,d,feats);
				
				//bigram
				
				if(i < s.words.size()-1){
					String str2 = c_word+"||"+s.words.get(i+1);
					add_one_str_feature(str2,d,feats);
				}
				
				/*
				//skip
				if(i < s.words.size()-2){
					String str2 = str+"[]"+s.words.get(i+2);
					add_one_str_feature(str2,d,feats);
				}
				*/
				
				//several phrase combinations
				/*
				String pc_pat[] = new String[]{"an","da","aa","na","dv"};
				if(i < size-1){
					for(String pat : pc_pat){
						String c_pos2 = s.pos.get(i+1);
						if(c_pos.length()>0 && c_pos2.length()>0)
							if(c_pos.charAt(0)==pat.charAt(0) && c_pos2.charAt(0)==pat.charAt(1))
								add_one_str_feature(c_word+s.words.get(i+1),d,feats);
					}
				}
				*/
				
				/* sentiment word
				boolean negation = false;
				if(c_word.startsWith(Sentence.NEG_HEAD)){
					c_word = c_word.substring(Sentence.NEG_HEAD.length());
					negation = true;
				}
				int sw = DictSentiment.dict_sentiment.search(c_word);
				if(negation){
					if(sw == DictSentiment.GOOD_SET)
						add_one_str_feature("B_"+c_word,d,feats);
					else if(sw == DictSentiment.BAD_SET)
						add_one_str_feature("G_"+c_word,d,feats);
				}
				else{
					if(sw == DictSentiment.GOOD_SET)
						add_one_str_feature("G_"+c_word,d,feats);
					else if(sw == DictSentiment.BAD_SET)
						add_one_str_feature("B_"+c_word,d,feats);
				}
				*/
			}
			
			
			deal_but(s);
			for(int i=0;i<size;i++){
				String c_pos = s.pos.get(i);
				String c_word = s.words.get(i);
				//unigram
				add_one_str_feature(c_word,d,feats);
				//bigram
				if(i < s.words.size()-1){
					String str2 = c_word+"||"+s.words.get(i+1);
					add_one_str_feature(str2,d,feats);
				}
			}
			
				
			//reverse
			/*
			for(int i=size-1;i>=0;i--){
				String c_pos = s.pos.get(i);
				String c_word = s.words.get(i);
				if(c_pos.length() > 0)
					if(c_pos.charAt(0)=='n'){
						add_one_str_feature("LASTN_"+c_word,d,feats);
						break;
					}
			}
			for(int i=size-1;i>=0;i--){
				String c_pos = s.pos.get(i);
				String c_word = s.words.get(i);
				if(c_pos.length() > 0)
					if(c_pos.charAt(0)=='v'){
						add_one_str_feature("LASTV_"+c_word,d,feats);
						break;
					}
			}
			for(int i=size-1;i>=0;i--){
				String c_pos = s.pos.get(i);
				String c_word = s.words.get(i);
				if(c_pos.length() > 0)
					if(c_pos.charAt(0)=='a'){
						add_one_str_feature("LASTA_"+c_word,d,feats);
						break;
					}
			}*/
			
		}
		return feats;
	}
}
