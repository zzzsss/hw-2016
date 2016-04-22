package task.dc.sentiment;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;

import nlpir.NLPIR.CLibrary;
import task.dc.DCCorpusReader;
import text.Paragraph;
import text.Tools;

public class SentimentReader2  extends DCCorpusReader{
	public List<Paragraph>[] read_corpus(String f){
		try{
			FileInputStream if_st = new FileInputStream(f);
			ObjectInputStream ioos = new ObjectInputStream(if_st);
			List<Paragraph>[] ret = (List<Paragraph>[])ioos.readObject();
			ioos.close();
			return ret;
		}
		catch (Exception e){
            e.printStackTrace();
            Error("Not obejct List<Paragraph>[]...");
		}
		return null;
	}
	
	public static void write_corpus(List<Paragraph>[] x,String s){
		try{
			FileOutputStream fos = new FileOutputStream(s);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(x);
			oos.close();
		}
		catch (Exception e){
            e.printStackTrace();
            Error("Not obejct List<Paragraph>[]...");
		}
	}
	
	//generate it
	public static void main(String []x){
		String f = "data/t_correct/data.obj";
		
		Tools.init(false,false);
		for(String xx : DictSentiment.dict_sentiment.get_bunch(DictSentiment.ALL_SET))
			CLibrary.Instance.NLPIR_AddUserWord(xx);
		List<Paragraph>[] d = (new SentimentReader1()).read_corpus("data/t_correct");
		
		write_corpus(d,f);
		
		
		SentimentReader2 r = new SentimentReader2();
		List<Paragraph>[] what = r.read_corpus(f);
		return ;
	}
}
