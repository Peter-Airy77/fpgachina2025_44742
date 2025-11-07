#ʱ��Լ��
create_clock -period 20.000 -name sys_clk [get_ports sys_clk]

#IO����Լ��
#------------------------------ϵͳʱ�Ӻ͸�λ-----------------------------------
set_property -dict {PACKAGE_PIN R4 IOSTANDARD LVCMOS15} [get_ports sys_clk]
set_property -dict {PACKAGE_PIN U7 IOSTANDARD LVCMOS15} [get_ports sys_rst_n]

#-----------------------------mux------------------------------
set_property IOSTANDARD LVCMOS33 [get_ports {row_sel[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {row_sel[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {row_sel[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {row_sel[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports {col_sel[0]}]              
set_property IOSTANDARD LVCMOS33 [get_ports {col_sel[1]}] 
set_property IOSTANDARD LVCMOS33 [get_ports {col_sel[2]}] 
set_property IOSTANDARD LVCMOS33 [get_ports {col_sel[3]}]
                                 
set_property PACKAGE_PIN Y16  [get_ports {row_sel[0]}]
set_property PACKAGE_PIN AB16 [get_ports {row_sel[1]}]
set_property PACKAGE_PIN AA15 [get_ports {row_sel[2]}]
set_property PACKAGE_PIN Y13  [get_ports {row_sel[3]}]

set_property PACKAGE_PIN AA16 [get_ports {col_sel[0]}]
set_property PACKAGE_PIN AB17  [get_ports {col_sel[1]}]
set_property PACKAGE_PIN AB15 [get_ports {col_sel[2]}]
set_property PACKAGE_PIN AA14 [get_ports {col_sel[3]}]


#---------------------------ADC------------------------------------
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[9]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[8]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_1[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[9]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[8]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ad_data_2[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports ad_clk_1]
set_property IOSTANDARD LVCMOS33 [get_ports ad_clk_2]
set_property PACKAGE_PIN C18 [get_ports ad_clk_1]
set_property PACKAGE_PIN P16 [get_ports ad_clk_2]
set_property PACKAGE_PIN E16 [get_ports {ad_data_1[0]}]
set_property PACKAGE_PIN D16 [get_ports {ad_data_1[1]}]
set_property PACKAGE_PIN D14 [get_ports {ad_data_1[2]}]
set_property PACKAGE_PIN D15 [get_ports {ad_data_1[3]}]
set_property PACKAGE_PIN B21 [get_ports {ad_data_1[4]}]
set_property PACKAGE_PIN A21 [get_ports {ad_data_1[5]}]
set_property PACKAGE_PIN B22 [get_ports {ad_data_1[6]}]
set_property PACKAGE_PIN C22 [get_ports {ad_data_1[7]}]
set_property PACKAGE_PIN D20 [get_ports {ad_data_1[8]}]
set_property PACKAGE_PIN C20 [get_ports {ad_data_1[9]}]
set_property PACKAGE_PIN B15 [get_ports {ad_data_2[0]}]
set_property PACKAGE_PIN B16 [get_ports {ad_data_2[1]}]
set_property PACKAGE_PIN C14 [get_ports {ad_data_2[2]}]
set_property PACKAGE_PIN C15 [get_ports {ad_data_2[3]}]
set_property PACKAGE_PIN C13 [get_ports {ad_data_2[4]}]
set_property PACKAGE_PIN B13 [get_ports {ad_data_2[5]}]
set_property PACKAGE_PIN AB18 [get_ports {ad_data_2[6]}]
set_property PACKAGE_PIN AA18 [get_ports {ad_data_2[7]}]
set_property PACKAGE_PIN U18 [get_ports {ad_data_2[8]}]
set_property PACKAGE_PIN U17 [get_ports {ad_data_2[9]}]
set_property PACKAGE_PIN A20 [get_ports ad_oe_1]
set_property PACKAGE_PIN V17 [get_ports ad_oe_2]
set_property PACKAGE_PIN B20 [get_ports ad_otr_1]
set_property PACKAGE_PIN W17 [get_ports ad_otr_2]
set_property IOSTANDARD LVCMOS33 [get_ports ad_oe_1]
set_property IOSTANDARD LVCMOS33 [get_ports ad_oe_2]
set_property IOSTANDARD LVCMOS33 [get_ports ad_otr_1]
set_property IOSTANDARD LVCMOS33 [get_ports ad_otr_2]

#-------------------------------uart------------------------
set_property -dict {PACKAGE_PIN D17 IOSTANDARD LVCMOS33} [get_ports uart_txd]

#-------------------------------LED Status Indicators------------------------
# LED pins for status monitoring
# NOTE: Please replace these pins with actual LED pins from your board's user manual
# These pins are chosen to be in a different bank from sys_clk (R4/Bank34)
# Using pins in the same bank as other LVCMOS33 signals
set_property -dict {PACKAGE_PIN T22 IOSTANDARD LVCMOS33} [get_ports led_fifo_overflow]
set_property -dict {PACKAGE_PIN T21 IOSTANDARD LVCMOS33} [get_ports led_frame_error]
set_property -dict {PACKAGE_PIN U22 IOSTANDARD LVCMOS33} [get_ports led_transmission_error]
set_property -dict {PACKAGE_PIN U21 IOSTANDARD LVCMOS33} [get_ports led_uart_tx_busy]