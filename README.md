# ClassicalChineseNER

文言文命名实体识别，基于BILSTM+CRF完成文言文的命名实体实体，识别实体包括人物、地点、机构、时间等。

嵌入层采用字词混合的表示方法，对于文言文，字采用自然单字，词采用jiayan工具进行分词，首先对词进行一层的LSTM，然后将输出和字的表示拼接在一期，通过一个BILSTM+CRF实现序列标注

参考甲言工具：
https://github.com/jiaeyan/Jiayan

标注数据来自于史记、明史等重要文献，标注样例如下：

屈原列传

![语料标注样例](https://github.com/chenking2020/ClassicalChineseNER/blob/main/images/data_sample.png)   

T1	Person 0 2	屈原
T2	Person 5 6	平
T3	Person 14 17	楚怀王
T4	Person 61 62	王
T5	Person 67 71	上官大夫
T8	Person 84 86	怀王
T9	Person 87 89	屈原
T6	Person 94 96	屈平
T7	Place 7 8	楚
T10	Person 102 106	上官大夫
T11	Person 112 114	屈平
T12	Person 123 124	王
T13	Person 125 127	屈平
T14	Person 158 159	王
T15	Person 162 164	屈平
T16	Person 166 168	屈平
T17	Person 169 170	王
T18	Person 270 272	屈平
T19	Person 312 314	屈平
T20	Person 362 364	帝喾
T21	Person 367 369	齐桓
T22	Person 372 373	汤
T23	Person 374 375	武
T24	Person 507 509	屈原
T25	Place 514 515	秦
T26	Place 517 518	齐
T27	Place 519 520	齐
T28	Place 521 522	楚
T29	Person 525 527	惠王
T30	Person 532 534	张仪
T31	Place 536 537	秦
T32	Place 543 544	楚
T33	Place 548 549	秦
T34	Place 551 552	齐
T35	Place 553 554	齐
T36	Place 555 556	楚
T37	Place 559 560	楚
T38	Place 563 564	齐
T39	Place 565 566	秦
T40	Place 568 569	商
T41	Place 570 571	於
T42	Person 578 581	楚怀王
T43	Person 584 586	张仪
T44	Place 589 590	齐
T45	Place 594 595	秦
T46	Person 598 600	张仪
T47	Person 605 606	仪
T48	Person 607 608	王
T49	Person 619 621	楚使
T50	Person 626 628	怀王
T51	Person 629 631	怀王
T52	Place 637 638	秦
T53	Place 639 640	秦
T54	Organization 647 649	楚师
T55	Place 650 651	丹
T56	Place 652 653	淅
T57	Person 662 664	屈匄
T58	Place 667 668	楚
T59	Place 669 671	汉中
T60	Person 673 675	怀王
T61	Place 686 687	秦
T62	Place 690 692	蓝田
T63	Place 693 694	魏
T64	Place 698 699	楚
T65	Place 700 701	邓
T66	Organization 702 704	楚兵
T67	Place 707 708	秦
T68	Place 711 712	齐
T69	Place 717 718	楚
T70	Place 719 720	楚
T71	Time 723 725	明年
T72	Place 726 727	秦
T73	Place 728 730	汉中
T74	Place 732 733	楚
T75	Person 736 738	楚王
T76	Person 748 750	张仪
