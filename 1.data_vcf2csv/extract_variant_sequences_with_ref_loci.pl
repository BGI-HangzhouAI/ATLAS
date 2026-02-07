#!/usr/bin/perl -w
use strict;
use warnings;

# Script: extract_variant_sequences_with_ref_loci.pl
# Purpose: 提取参考基因组区域 + VCF 变异信息，生成每个样本的两个单倍型序列（支持正负链反向互补），和对应参考基因组上的位置，输出为 CSV 文件
# Author: Haiqiang Zhang, BGI-Research
# Date: 2025-08-27
# Usage: perl extract_variant_sequences.pl <VCF> <"chr:start-end:+/-;chr:start-end:+/-"> <ref_genome.fa> <out_prefix>

if (@ARGV<4){
   warn "perl $0 <VCF> <chr:start-end:+/-;chr:start-end:+/-(0-based start)> <ref_genome.fa> <out_prefix>\n";
   exit 1;
   }

my $vcf_file=shift;
my $chr_locis=shift;
my $fa=shift; #human genome
my $out_pfx=shift;
my $target_seq;
my %hg;
open (my $HG,'<', $fa) or die "Could not open file '$fa' $!";
my $S="";
while (<$HG>)
{
        chomp;
        if ( /^>/ )
        {
                $_ =~ s/>//g;
                my @fields=split;
                $S= $fields[0];
                $hg{$S}="";
                #print "processing $S\t";
        }
        else
        {
                my $str=uc $_;   #transform all lowcase to uppercase
                $hg{$S} .= $str ;
        }
}
close $HG;

#chr1   564760  564881  121     0       +       TA      CT
#chr1   567478  567543  65      0       +       TA  CT
# extract genome seq for each target region

my %hash_ref_seq;
my %hash_chr_record;
my @P=split/;/,$chr_locis;
my %hash_loci;
#my @SEQ;
foreach my $chr_loci(@P)
{
  if ($chr_loci=~/^(\S+)\:(\d+)\-(\d+)/)
  {
   my ($chr, $start, $end) = ($1, $2, $3);
   my $len=$end-$start;
   if (exists $hash_chr_record{$chr}){           #collect all target regions in the same chromosome
      $hash_chr_record{$chr}.="\t".$chr_loci;
   }
   else{
      $hash_chr_record{$chr}=$chr_loci;
   }
   $target_seq=substr($hg{$chr},$start,$len);
   $hash_ref_seq{$chr_loci}=$target_seq;
   $hash_loci{$chr_loci}=[$start+1..$end];            ##store all loci in the target region (1-based)
  }
}

#read vcf files and extract VCF records in target regions to reduce the memory
my %hash_ref;
my %hash_var;
my $hap_a;
my $hap_b;
my @SAMP;
my %hash_vcf_loc;
my $sample_header_line;
my %vcf_by_region;

# 使用新的函数打开VCF文件
my $vcf_fh = open_vcf($vcf_file);

while (<$vcf_fh>)
{
   chomp;
   my $line=$_;
   next if ($line=~/^\#\#/);
   my @F=split/\t/,$line;
      if ($F[0]=~/^\#/)
      {
         #CHROM  POS     ID         REF     ALT     QUAL    FILTER  INFO    FORMAT  D20     D21     D22     D23     D24
         @SAMP=@F[9..$#F];
         $sample_header_line=$line;
#         print OUT, $line;
      }
      else
      {
            if (exists $hash_chr_record{$F[0]})
            {
               my @M=split/\t/,$hash_chr_record{$F[0]};
               foreach my $m(@M)
               {
                 if ($m=~/^(\S+)\:(\d+)\-(\d+)/)
                 {
                  my $l_chr=$1;
                  my $l_start=$2;
                  my $l_end=$3;
                  if ($F[1]>$l_start && $F[1]<=$l_end)
                  {
                     $hash_vcf_loc{$F[0]."-".$F[1]}=$m;
                     push @{$vcf_by_region{$m}}, $line;     #store vcf records by targets region
                  }
                 }
               }
            }
         }
}
close $vcf_fh;


my %hash_sample_seq;
my %hash_sample_loc;
foreach my $sample (@SAMP)
{
  foreach my $loc(@P)
  {
    $hash_sample_seq{$sample}{$loc}{"m"}=$hash_ref_seq{$loc};
    $hash_sample_seq{$sample}{$loc}{"n"}=$hash_ref_seq{$loc};
    $hash_sample_loc{$sample}{$loc}{"m"}= [@{$hash_loci{$loc}}];   ##initialize the loci array for each sample and each hapolotype
    $hash_sample_loc{$sample}{$loc}{"n"}= [@{$hash_loci{$loc}}];
  }
}

# 对每个目标区域内的变异按位置降序排序
foreach my $region (keys %vcf_by_region) {
    @{$vcf_by_region{$region}} = sort {
        my @a_fields = split /\t/, $a;
        my @b_fields = split /\t/, $b;
        $b_fields[1] <=> $a_fields[1];  # 位置降序
    } @{$vcf_by_region{$region}};
}


open OUT, ">${out_pfx}_target.rev.sorted.vcf";
print OUT $sample_header_line."\n";
foreach my $region (@P) {  # 按原始区域顺序处理
    next unless exists $vcf_by_region{$region};
    
    foreach my $line (@{$vcf_by_region{$region}}){
        print OUT $line."\n";
    }
    undef @{$vcf_by_region{$region}}; #delete array to release memory
   }
close OUT;



open IN, "${out_pfx}_target.rev.sorted.vcf" or die "Could not open file: $!";
 while (<IN>)
 {
    chomp;
    my $line=$_;
    my @F=split/\t/,$line;

    #CHROM  POS     ID         REF     ALT     QUAL    FILTER  INFO    FORMAT  D20     D21     D22     D23     D24
    #chr1   12938 rs756849893  GCAAA   G       .       .       .       GT      0|0     0|0     0|0     0|0     0|0
    #chr1   13273 rs531730856  G       C       .       .       .       GT      0|0     0|1     0|0     0|0     0|0
    
    next if ($F[0]=~/^\#/);
    my $ref_alle=$F[3];
    my $ref_len=length($ref_alle);
    my @ALT=split/,/,$F[4];
    my $val_loci_idx;               #index of loci in ref req
    if ($hash_vcf_loc{$F[0]."-".$F[1]}=~/^(\S+)\:(\d+)\-(\d+)/)
    {
       $val_loci_idx=$F[1]-$2-1;
    }

    for my $i (9..$#F)
    {
      my $sample=$SAMP[$i-9];
      if ($F[$i]=~/^(\d+)\|(\d+)/)
      {
         my $hap_a=$1;
         my $hap_b=$2;
         if ($hap_a != 0)
         {
            my $alt_len_m=length($ALT[$hap_a-1]);     ##obtain the length of alt allele in hap 1
            substr($hash_sample_seq{$sample}{$hash_vcf_loc{$F[0]."-".$F[1]}}{"m"}, $val_loci_idx, $ref_len)=$ALT[$hap_a-1]; #replace the ref with alt in hap 1
            splice(@{$hash_sample_loc{$sample}{$hash_vcf_loc{$F[0]."-".$F[1]}}{"m"}}, $val_loci_idx, $ref_len, ($F[1]) x $alt_len_m);  ##update the loci array for hap 1
         }
         if ($hap_b != 0)
         {
            my $alt_len_m=length($ALT[$hap_b-1]);     ##obtain the length of alt allele in hap 1
            substr($hash_sample_seq{$sample}{$hash_vcf_loc{$F[0]."-".$F[1]}}{"n"}, $val_loci_idx, $ref_len)=$ALT[$hap_b-1]; #replace the ref with alt in hap 2
            splice(@{$hash_sample_loc{$sample}{$hash_vcf_loc{$F[0]."-".$F[1]}}{"n"}}, $val_loci_idx, $ref_len, ($F[1]) x $alt_len_m);  ##update the loci array for hap 2
         }
      }     
    }
 }
   
####################for testing
#print join(",", $SAMP[0], $P[0].":m:".$hash_ref_seq{$P[0]}, $SAMP[0], $P[0].":n:".$hash_ref_seq{$P[0]})."\n";

foreach my $sample(@SAMP)
{
   my @CSV=($sample);
   foreach my $position(@P){
      for my $i ("m", "n")
      {
         my $seq=$hash_sample_seq{$sample}{$position}{$i};
         my @loci_array=@{$hash_sample_loc{$sample}{$position}{$i}};
         if ($position=~/^\S+\:\S+\:([+-])$/)
         {
           my $strand=$1;
           if ($strand eq "-")
           {
              $seq=revcomp($hash_sample_seq{$sample}{$position}{$i});
              @loci_array=reverse @loci_array;    ##reverse the loci array when the strand is reverse strand
           }
         }
         my $loci_string=join(",", @loci_array);         ##loci array
         push @CSV, join(":", $position, $i, $seq, '"'.$loci_string.'"');
       }
      }
   print join (",",@CSV)."\n";
}

# 添加函数来智能打开VCF文件（支持压缩和非压缩）
sub open_vcf {
    my ($filename) = @_;

    my $fh;

    
    if ($filename =~ /\.gz$/i) {
        # 压缩文件，使用zcat或gunzip
        if (system("which zcat >/dev/null 2>&1") == 0) {
            open($fh, "zcat $filename |") or die "Cannot open compressed file $filename: $!";
        } elsif (system("which gunzip >/dev/null 2>&1") == 0) {
            open($fh, "gunzip -c $filename |") or die "Cannot open compressed file $filename: $!";
        } else {

            die "Error: Neither zcat nor gunzip found. Cannot read compressed file $filename\n";
        }
    } else {
        # 普通文件

        open($fh, '<', $filename) or die "Cannot open file $filename: $!";

    }
    
    return $fh;
}

##reverse complement function
sub revcomp {
    my ($seq) = @_;
    # Convert to capital letters
    $seq = uc $seq;

    # Replace all non-ATGCN characters with N (optional to enhance robustness)
    $seq =~ s/[^ATGC]/N/g;

    # reverse complement
    $seq = reverse $seq;
    $seq =~ tr/ATGC/TACG/;
    return $seq;
}