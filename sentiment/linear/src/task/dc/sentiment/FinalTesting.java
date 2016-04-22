package task.dc.sentiment;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.util.*;
import java.util.regex.*;

import task.dc.DCCorpusReader;
import text.Paragraph;

//for final testing
public class FinalTesting {
	static List<String> get_sentences(String f){
		List<String> sentences = new ArrayList<String>();
		try{
			//read xml style --- but don't check format
			FileInputStream in = new FileInputStream(f);
			BufferedReader dr = new BufferedReader(new InputStreamReader(in,"UTF-8"));
			StringBuilder temp = new StringBuilder();
			String line="";
			int state = 0;	//out
			Pattern p = Pattern.compile("\\s*<weibo.*>(.*)</weibo>\\s*");
			while((line=dr.readLine()) != null){
				Matcher m = p.matcher(line);
				if(m.matches()){
					sentences.add(m.group(1));
				}
			}
			dr.close();
		}catch (Exception e){
            e.printStackTrace();
		}
		return sentences;
	}
	
	static class DCCorpusReader_test extends DCCorpusReader{
		public List<Paragraph>[] read_corpus(String f){
			List<Paragraph>[] ret = new List[1];
			ret[0] = new ArrayList<Paragraph>();
			List<String> sentences = FinalTesting.get_sentences(f);
			for(String s : sentences)
				ret[0].add(new Paragraph(s));
			return ret;
		}
	}
	public static void eval(String origin_f,List<Double> res,String output_file){
		try{
			List<String> sentences = FinalTesting.get_sentences(origin_f);
			OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(output_file),"UTF-8");
			out.write("<weibos>\n");
			for(int i=0;i<sentences.size();i++){
				int fill = (res.get(i)>0.5) ? 1 : -1;
				out.write("\t<weibo id=\""+(i+1)+"\" polarity=\""+fill+"\">"+sentences.get(i)+"</weibo>\n");
			}
			out.write("</weibos>\n");
			out.close();
			
		}catch (Exception e){
            e.printStackTrace();
        }
	}
}	
