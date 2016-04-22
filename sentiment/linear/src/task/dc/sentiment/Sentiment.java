package task.dc.sentiment;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import nlpir.NLPIR.CLibrary;
import task.dc.*;
import text.*;
import training.*;

public class Sentiment {
	static int NB=0,SVMC=1,SVMR=2,SVM_LI=3;
	
	static DCCorpusReader r = new SentimentReader2();
	//static DCFeatureGenerator f = new FGNormal(new SentimentFeature4());
	//static DCFeatureGenerator f = new FGSelection(new SentimentFeature3(),4,300);
	static DCFeatureGenerator f = new FGDelf(new SentimentFeature4(),"testing2/tf.score");
	static Mach m;
	static Dict d;
	static DCClassifyTask task;
	static{
		Tools.init(false,false);
		for(String x : DictSentiment.dict_sentiment.get_bunch(DictSentiment.ALL_SET))
			CLibrary.Instance.NLPIR_AddUserWord(x);
	}

	//init the task
	static void init_task(int mach){
		if(mach == NB)
			m = new NaiveBayes();
		else
			m = new LibsvmCommon(mach-1,false);
		task = new DCClassifyTask(r,f,m);
	}
	static void init_task(String mach,String dict){
		m = Mach.read(new File(mach));
		d = new Dict(new File(dict));
		task = new DCClassifyTask(r,f,m,d);
	}
	
	static void final_demo() throws Exception{
		f = new FGNormal(new SentimentFeature4());
		init_task("testing2/f4_n.mach","testing2/f4_n.fdict");
		
		BufferedInputStream i = new BufferedInputStream(System.in);
        BufferedReader br = new BufferedReader(new InputStreamReader(i));
        do{
        	System.out.print("Enter sentence: ");
        	String line = br.readLine();
        	if(line == null)
        		break;
        	double ret = task.test_one(line);
        	if(ret>0.5)
        		System.out.println("Thanks a lot...");
        	else
        		System.out.println("Sorry to hear that...");
        }while(true);
	}
	
	static void final_train(){
		//f = new FGDelf(new SentimentFeature4(),"testing2/tf.score");
		//init_task(3);
		
		f = new FGNormal(new SentimentFeature1());
		init_task(0);
		
		//task.train_cv("data/t_correct/data.obj", "testing2/nothing");
		//task.train_part("data/t_correct/data.obj", "testing2/nothing");
		task.train_all("data/t_correct/data.obj", "testing2/ftest");
		return;
	}
	
	static class DCCorpusReader_test extends DCCorpusReader{
		String xml_tag;
		public DCCorpusReader_test(String tag){
			xml_tag = tag;
		}
		public List<Paragraph>[] read_corpus(String f){
			//read one file
			List<Paragraph>[] ret = new List[1];
			ret[0] = new ArrayList<Paragraph>();
			try{
				//read xml style --- but don't check format
				FileInputStream in = new FileInputStream(f);
				BufferedReader dr = new BufferedReader(new InputStreamReader(in,"UTF-8"));
				StringBuilder temp = new StringBuilder();
				String line="";
				while((line=dr.readLine()) != null){
					if(!line.isEmpty()){
						if(line.indexOf("<"+xml_tag) > -1)
							;
						else if(line.indexOf("</"+xml_tag) > -1){
							ret[0].add(new Paragraph(temp.toString()));
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
			return ret;
		}
	}
	static void final_test(String filename,String output_name){
		//read in test data
		f = new FGDelf(new SentimentFeature4(),"testing2/f4_tf.score");
		r = new DCCorpusReader_test("review");
		init_task("testing2/f4_tf.mach","testing2/f4_tf.fdict");
		List<Double> [] result = task.test_all(filename,"WHAT");
		(new Evaluator("review","label")).eval(filename,result[0]);
	}

	static void finalfinal_test(String filename,String output_name){
		r = new FinalTesting.DCCorpusReader_test();
		f = new FGDelf(new SentimentFeature4(),"testing2/tf.score");
		init_task("testing2/f4_tf.mach","testing2/f4_tf.fdict");
		List<Double> [] result = task.test_all(filename,"WHAT");
		FinalTesting.eval(filename, result[0], output_name);
	}
	public static void main(String[] x) throws Exception{
		final_demo();
		//final_test("data/test/test.label.cn.txt","WHAT");
		//final_train();
		//finalfinal_test("data/test/task2input.xml","testing2/test.out");
	}
}
