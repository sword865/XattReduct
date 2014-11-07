package Xreducer_fuzzy;

public class SStyle_MaxMin extends SimilarityStyle{
	public double getSimilarityValue(double x, double y){
		//super.getSimilarityValue(x, y);
		if(Double.isNaN(x)||Double.isNaN(y))
			return 1;
		if(Vstdvar==0)
			return 1;
		else
		{
			double part1 = (y-(x-Vstdvar))/(Vstdvar);
			double part2 = ((x+Vstdvar)-y)/(Vstdvar);
			return Math.max(Math.min(part1,part2), 0);
		}
			
	}
	public String getInformation(){
		String str = "Similarity measure: max(min( (a(y)-(a(x)-sigma_a)) / (a(x)-(a(x)-sigma_a)),((a(x)+sigma_a)-a(y)) / ((a(x)+sigma_a)-a(x)) , 0).";
		System.out.println(str);
		return str;
		}
	public String getInfor(){
		return "MaxMin";
	}
}
