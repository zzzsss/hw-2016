package text;

import java.util.*;
import java.io.*;

public class Paragraph implements Serializable{
	//break the sentences in this simple way for chinese
	static String[] END_INDICATE_ARRAY = new String[]{"!","?","£¿","£¡","¡£"};
	static HashSet<String> END_INDICATE = new HashSet<String>();
	static{
		for(String x:END_INDICATE_ARRAY)
			END_INDICATE.add(x);
	}
	
	public List<Sentence> sents;
	private void self_init(String s){
		sents = new ArrayList<Sentence>();
		StringBuilder x = new StringBuilder();
		for(int i=0;i<s.length();i++){
			char t = s.charAt(i);
			if(Character.isSpaceChar(t))
				continue;
			x.append(t);
			if(END_INDICATE.contains(t+"")){
				sents.add(new Sentence(x.toString()));
				x =  new StringBuilder();
			}
		}
		if(!x.toString().isEmpty()){
			sents.add(new Sentence(x.toString()));
		}
	}
	
	//-----------
	public Paragraph(String s){
		self_init(s);
	}
	public Paragraph(File f){
		StringBuilder x = new StringBuilder();
		try{
			FileInputStream in = new FileInputStream(f);
			BufferedReader dr = new BufferedReader(new InputStreamReader(in));
			String line="";
			while((line=dr.readLine()) != null){
				x.append(line);
			}
			dr.close();
		}catch (Exception e){
            e.printStackTrace();
		}
		self_init(x.toString());
	}
	public String toString(){
		String x = "";
		for(Sentence s : sents)
			x += s;
		return x;
	}
}
