package task.dc.sentiment;

import java.util.*;

import task.dc.*;
import text.*;

import java.io.*;

public class SentimentReader1 extends DCCorpusReader{
	static String classes_fname = "c";
	
	int mode;
	//----
	
	private List<Paragraph>[] read_seperate_dirs(int classes,HashMap<String,File> fnames){
		//for all those classes
		List<Paragraph>[] ret = new List[classes];
		for(int i=0;i<classes;i++){
			ret[i] = new ArrayList<Paragraph>();
			String name = classes_fname+i;
			File ff = fnames.get(name);
			if(!ff.exists() || !ff.isDirectory())
				Error("No subdir "+ff.toString());
			File suf_all[] = ff.listFiles();
			System.out.println("-- Reading class "+i+" : "+suf_all.length+" files.");
			for(File subf : suf_all)
				ret[i].add(new Paragraph(subf));
		}
		return ret;
	}
	private List<Paragraph>[] read_seperate_files(int classes,HashMap<String,File> fnames){
		//for all those classes
		List<Paragraph>[] ret = new List[classes];
		for(int i=0;i<classes;i++){
			ret[i] = new ArrayList<Paragraph>();
			String name = classes_fname+i;
			File ff = fnames.get(name);
			System.out.println("-- Reading class "+i);
			//read one piece of file
			try{
				//read xml style --- but don't check format
				FileInputStream in = new FileInputStream(ff);
				BufferedReader dr = new BufferedReader(new InputStreamReader(in,"UTF-8"));
				StringBuilder temp = new StringBuilder();
				String line="";
				while((line=dr.readLine()) != null){
					if(!line.isEmpty()){
						if(line.indexOf("<review") > -1)
							;
						else if(line.indexOf("</review") > -1){
							ret[i].add(new Paragraph(temp.toString()));
							temp = new StringBuilder();
						}
						else
							temp.append(line);
					}
				}
				dr.close();
			}catch (Exception e){
	            e.printStackTrace();
	            Error("WHAT??");
			}
		}
		return ret;
	}
	
	//----------------
	public SentimentReader1(int m){
		mode = m;
	}
	public SentimentReader1(){
		mode = 1;
	}
	public List<Paragraph>[] read_corpus(String f){
		//specified file and dir format
		int classes = 0;
		File folder = new File(f);
		if(!folder.exists() || !folder.isDirectory())
			Error("No dir "+f);
		//how many classes
		File[] files = folder.listFiles();
		HashMap<String,File> fnames = new HashMap<String,File>();
		for(File ff : files)
			fnames.put(ff.getName(),ff);
		for(classes=0;;classes++){
			String name = classes_fname+classes;
			if(!fnames.containsKey(name))
				break;
		}
		if(classes < 2)	//no meaning for classification
			Error("Classes < 2.");
		
		switch(mode){
		case 0:	
			return read_seperate_dirs(classes,fnames);
		case 1:
			return read_seperate_files(classes,fnames);
		default:
			return null;
		}
	}
}
