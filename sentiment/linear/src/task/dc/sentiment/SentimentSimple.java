package task.dc.sentiment;

import java.util.*;
import text.*;
import task.dc.FeatureCounter;

public class SentimentSimple {
	static Random rand;
	static{
		rand = new Random();
	}
	//just for checking --- based on dict(non statistic) and random <just for fun...>
	static void check_dict(List<Paragraph>[] data,DictSentiment d){
		//2 sets: bad vs. good
		int total = 0;
		int acc = 0;
		for(int i=0;i<2;i++){
			total += data[i].size();
			for(Paragraph p : data[i]){
				int good = 0,bad = 0;
				for(Sentence s : p.sents){
					//first deal with negation
					FeatureCounter.negation(s);
					for(String x : s.words){
						String x_really = x;
						if(x.startsWith(FeatureCounter.NEG_HEAD)){
							x_really = x_really.substring(FeatureCounter.NEG_HEAD.length());
						}
						if(d.search(x_really) == DictSentiment.BAD_SET){
							if(x.startsWith(FeatureCounter.NEG_HEAD))
								good++;
							else
								bad++;
						}
						else if(d.search(x_really) == DictSentiment.GOOD_SET){
							if(x.startsWith(FeatureCounter.NEG_HEAD))
								bad++;
							else
								good++;
						}
					}
				}
				int inference;
				if(good > bad)
					inference = 1;
				else if(good < bad)
					inference = 0;
				else
					inference = rand.nextInt(1);
				if(i == inference)
					acc++;
			}
		}
		System.out.println("Accuracy for this simple method is: "+acc/(double)total);
	}
	
	public static void main(String[] s){
		DictSentiment d = DictSentiment.dict_sentiment;
		List<Paragraph>[] data = (new SentimentReader2()).read_corpus("data/t_correct/data.obj");
		check_dict(data,d);
	}
}
