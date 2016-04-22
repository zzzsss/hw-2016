package task.dc;

import java.util.*;

import text.Dict;
import text.Paragraph;
import text.Sentence;

public abstract class FeatureCounter {
	abstract public HashMap<Integer,Integer> get_one(Paragraph p,Dict d);
	
	//count the features
	public List<HashMap<Integer,Integer>>[] count(List<Paragraph>[] pl,Dict d){
		List<HashMap<Integer,Integer>>[] ret = new List[pl.length];
		for(int i=0;i<pl.length;i++){
			ret[i] = new ArrayList<HashMap<Integer,Integer>>();
			for(Paragraph p: pl[i])
				ret[i].add(get_one(p,d));
		}
		return ret;
	}
	
	protected static void add_one_str_feature(String x,Dict d,HashMap<Integer,Integer> feats){
		int ind = d.add(x);
		if(ind >= 0){
			Integer ori = feats.get(ind);
			if(ori == null)
				feats.put(ind, 1);
			else
				feats.put(ind, ori+1);
		}
	}
	
	// although-but another routine like negation
		//--maybe this should be added to the Sentence class, but since it is not convenient to change...
		static String[] ALTHOUGH_INDICATE_ARRAY = new String[]{"虽说","固然","非但","虽然","尽管"};
		static HashSet<String> ALTHOUGH_INDICATE = new HashSet<String>();
		static String[] BUT_INDICATE_ARRAY = new String[]{"不过","但","但是","而是","反之","可是","然而","转而","恰恰相反","反倒","反而","却","仍"};
		static HashSet<String> BUT_INDICATE = new HashSet<String>();
		static{
			for(String x:ALTHOUGH_INDICATE_ARRAY)
				ALTHOUGH_INDICATE.add(x);
			for(String x:BUT_INDICATE_ARRAY)
				BUT_INDICATE.add(x);
		}
		static String AB_HEAD[] = new String[]{"ALTHOUGH_","BUT_"};
		protected static void deal_but(Sentence s){
			//simple method
			int idi = -1;	//-1 nothing, 0 although, 1 but
			for(int i=0;i<s.words.size();i++){
				String now = s.words.get(i);
				if(ALTHOUGH_INDICATE.contains(now))
					idi = 0;
				else if(BUT_INDICATE.contains(now))
					idi = 1;
				else if(s.pos.get(i).length()>0 && s.pos.get(i).charAt(0) == 'w')	//punct
					idi = -1;
				else if(idi >= 0){
					//if(s.pos.get(i).length()>0){ 
					//	char x = s.pos.get(i).charAt(0);
					//	if(x=='v' || x=='n' || x=='a')
							s.words.set(i,AB_HEAD[idi]+now);
					//}
				}
			}
		}
		
		//negation --- for sentiment
		static String[] NEG_INDICATE_ARRAY = new String[]{"不","没有","不是","没","不如"};
		static HashSet<String> NEG_INDICATE = new HashSet<String>();
		static String[] STOP_INDICATE_ARRAY = new String[]{"!","?",",",".","？","！","，","。"};
		static HashSet<String> STOP_INDICATE = new HashSet<String>();
		static{
			for(String x:NEG_INDICATE_ARRAY)
				NEG_INDICATE.add(x);
			for(String x:STOP_INDICATE_ARRAY)
				STOP_INDICATE.add(x);
		}
		public static String NEG_HEAD = "NOT_";
		public static void negation(Sentence s){
			//simple method
			int neg = 0;
			for(int i=0;i<s.words.size();i++){
				String now = s.words.get(i);
				if(NEG_INDICATE.contains(now))
					neg = 1-neg;
				else if(STOP_INDICATE.contains(now))
					neg = 0;
				else if(neg==1)
					s.words.set(i,NEG_HEAD+now);
			}
		}
}
