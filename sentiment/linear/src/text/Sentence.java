package text;

import java.util.*;

public class Sentence implements java.io.Serializable{
	//1.nlpir
	public List<String> words;
	public List<String> pos;
	public String segmented_str;
	/*
	//2.tagger
	public List<String> pos2;	//using stanford pos tagger
	//3.ner
	List<String> ner;
	//4.parser
	Tree tree;
	List<TypedDependency> dep;
	*/
	
	//constructor
	//-- x is the extra info for seg(if no info then null)
	protected void self_init(String s,String[] x){
		Object[] sseg = Tools.get_seg(s,x);
		words = (List<String>)sseg[0];
		pos = (List<String>)sseg[1];
		segmented_str = (String)sseg[2];
		/*
		pos2 = Tools.get_pos(words);
		//options
		ner = Tools.get_ner(segmented_str);
		Object[] sparse = Tools.get_parse(words);
		if(sparse != null){
			tree = (Tree)(sparse[0]);
			dep = (List<TypedDependency>)sparse[1];
		}
		*/
	}
	public Sentence(String s,String[] x){	
		self_init(s,x);
	}
	public Sentence(String s){
		self_init(s,null);
	}
	public String toString(){
		return words.toString();
	}
	
	//bag of words
	public List<Integer> get_bagofwords(){
		List<Integer> ret = new ArrayList<Integer>();
		return ret;
	}
}
