#pragma once
#include "ParallelAlgorithm.h"
#include <string>
namespace GraduateWork {

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

public ref class MyForm : public System::Windows::Forms::Form
{
public:
MyForm(void)
{
InitializeComponent();
}

protected:
~MyForm()
{
if (components)
{
delete components;
}
}
private: System::Windows::Forms::TabControl^ tabControl1;
protected:
private: System::Windows::Forms::TabPage^ tabPage1;
private: System::Windows::Forms::TabPage^ tabPage2;
private: System::Windows::Forms::Panel^ panel2;
private: System::Windows::Forms::Panel^ panel1;
private: System::Windows::Forms::Label^ label1;
private: System::Windows::Forms::PictureBox^ pictureBox1;
private: System::Windows::Forms::GroupBox^ groupBox1;
private: System::Windows::Forms::Button^ button2;
private: System::Windows::Forms::Button^ button1;
private: System::Windows::Forms::NumericUpDown^ numericUpDown5;
private: System::Windows::Forms::Label^ label5;
private: System::Windows::Forms::NumericUpDown^ numericUpDown4;
private: System::Windows::Forms::NumericUpDown^ numericUpDown3;
private: System::Windows::Forms::NumericUpDown^ numericUpDown2;
private: System::Windows::Forms::NumericUpDown^ numericUpDown1;
private: System::Windows::Forms::Label^ label4;
private: System::Windows::Forms::Label^ label3;
private: System::Windows::Forms::Label^ label2;
private: System::Windows::Forms::Label^ label6;
private: System::Windows::Forms::Label^ label10;
private: System::Windows::Forms::Label^ label9;
private: System::Windows::Forms::Label^ label8;
private: System::Windows::Forms::Label^ label7;
private: System::Windows::Forms::GroupBox^ groupBox4;
private: System::Windows::Forms::GroupBox^ groupBox3;
private: System::Windows::Forms::DataGridView^ dataGridView1;
private: System::Windows::Forms::GroupBox^ groupBox2;
private: System::Windows::Forms::Label^ label11;
private: System::Windows::Forms::Label^ label12;
private: System::Windows::Forms::Label^ label13;
private: System::Windows::Forms::Label^ label14;
private: System::Windows::Forms::Label^ label15;
private: System::Windows::Forms::DataGridView^ dataGridView2;
private: System::Windows::Forms::Label^ label20;
private: System::Windows::Forms::Label^ label19;
private: System::Windows::Forms::Label^ label18;
private: System::Windows::Forms::Label^ label17;
private: System::Windows::Forms::Label^ label16;
private: System::Windows::Forms::Label^ label25;
private: System::Windows::Forms::Label^ label24;
private: System::Windows::Forms::Label^ label23;
private: System::Windows::Forms::Label^ label22;
private: System::Windows::Forms::Label^ label21;

private:
System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
void InitializeComponent(void)
{
System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(MyForm::typeid));
this->tabControl1 = (gcnew System::Windows::Forms::TabControl());
this->tabPage1 = (gcnew System::Windows::Forms::TabPage());
this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
this->dataGridView2 = (gcnew System::Windows::Forms::DataGridView());
this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
this->dataGridView1 = (gcnew System::Windows::Forms::DataGridView());
this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
this->label20 = (gcnew System::Windows::Forms::Label());
this->label19 = (gcnew System::Windows::Forms::Label());
this->label18 = (gcnew System::Windows::Forms::Label());
this->label17 = (gcnew System::Windows::Forms::Label());
this->label16 = (gcnew System::Windows::Forms::Label());
this->label10 = (gcnew System::Windows::Forms::Label());
this->label9 = (gcnew System::Windows::Forms::Label());
this->label8 = (gcnew System::Windows::Forms::Label());
this->label7 = (gcnew System::Windows::Forms::Label());
this->label6 = (gcnew System::Windows::Forms::Label());
this->panel2 = (gcnew System::Windows::Forms::Panel());
this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
this->label25 = (gcnew System::Windows::Forms::Label());
this->label24 = (gcnew System::Windows::Forms::Label());
this->label23 = (gcnew System::Windows::Forms::Label());
this->label22 = (gcnew System::Windows::Forms::Label());
this->label21 = (gcnew System::Windows::Forms::Label());
this->label11 = (gcnew System::Windows::Forms::Label());
this->label12 = (gcnew System::Windows::Forms::Label());
this->label13 = (gcnew System::Windows::Forms::Label());
this->label14 = (gcnew System::Windows::Forms::Label());
this->label15 = (gcnew System::Windows::Forms::Label());
this->panel1 = (gcnew System::Windows::Forms::Panel());
this->button2 = (gcnew System::Windows::Forms::Button());
this->button1 = (gcnew System::Windows::Forms::Button());
this->numericUpDown5 = (gcnew System::Windows::Forms::NumericUpDown());
this->label5 = (gcnew System::Windows::Forms::Label());
this->numericUpDown4 = (gcnew System::Windows::Forms::NumericUpDown());
this->numericUpDown3 = (gcnew System::Windows::Forms::NumericUpDown());
this->numericUpDown2 = (gcnew System::Windows::Forms::NumericUpDown());
this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
this->label4 = (gcnew System::Windows::Forms::Label());
this->label3 = (gcnew System::Windows::Forms::Label());
this->label2 = (gcnew System::Windows::Forms::Label());
this->label1 = (gcnew System::Windows::Forms::Label());
this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
this->tabPage2 = (gcnew System::Windows::Forms::TabPage());
this->tabControl1->SuspendLayout();
this->tabPage1->SuspendLayout();
this->groupBox4->SuspendLayout();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->dataGridView2))->BeginInit();
this->groupBox3->SuspendLayout();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->dataGridView1))->BeginInit();
this->groupBox1->SuspendLayout();
this->panel2->SuspendLayout();
this->groupBox2->SuspendLayout();
this->panel1->SuspendLayout();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown5))->BeginInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->BeginInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->BeginInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->BeginInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->BeginInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
this->SuspendLayout();
this->tabControl1->Controls->Add(this->tabPage1);
this->tabControl1->Controls->Add(this->tabPage2);
this->tabControl1->Location = System::Drawing::Point(-5, 1);
this->tabControl1->Name = L"tabControl1";
this->tabControl1->SelectedIndex = 0;
this->tabControl1->Size = System::Drawing::Size(1277, 757);
this->tabControl1->TabIndex = 0;
this->tabPage1->BackColor = System::Drawing::SystemColors::Control;
this->tabPage1->Controls->Add(this->groupBox4);
this->tabPage1->Controls->Add(this->groupBox3);
this->tabPage1->Controls->Add(this->groupBox1);
this->tabPage1->Controls->Add(this->panel2);
this->tabPage1->Controls->Add(this->panel1);
this->tabPage1->Controls->Add(this->pictureBox1);
this->tabPage1->Location = System::Drawing::Point(4, 22);
this->tabPage1->Name = L"tabPage1";
this->tabPage1->Padding = System::Windows::Forms::Padding(3);
this->tabPage1->Size = System::Drawing::Size(1269, 731);
this->tabPage1->TabIndex = 0;
this->tabPage1->Text = L" ";
this->tabPage1->Click += gcnew System::EventHandler(this, &MyForm::tabPage1_Click);
this->groupBox4->BackColor = System::Drawing::SystemColors::ActiveCaption;
this->groupBox4->Controls->Add(this->dataGridView2);
this->groupBox4->Font = (gcnew System::Drawing::Font(L"Georgia", 10, System::Drawing::FontStyle::Bold));
this->groupBox4->Location = System::Drawing::Point(402, 437);
this->groupBox4->Name = L"groupBox4";
this->groupBox4->Size = System::Drawing::Size(856, 291);
this->groupBox4->TabIndex = 6;
this->groupBox4->TabStop = false;
this->groupBox4->Text = L"   ";
this->dataGridView2->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
this->dataGridView2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->dataGridView2->Location = System::Drawing::Point(13, 31);
this->dataGridView2->Name = L"dataGridView2";
this->dataGridView2->Size = System::Drawing::Size(840, 253);
this->dataGridView2->TabIndex = 1;
this->groupBox3->BackColor = System::Drawing::Color::RosyBrown;
this->groupBox3->Controls->Add(this->dataGridView1);
this->groupBox3->Font = (gcnew System::Drawing::Font(L"Georgia", 10, System::Drawing::FontStyle::Bold));
this->groupBox3->Location = System::Drawing::Point(402, 138);
this->groupBox3->Name = L"groupBox3";
this->groupBox3->Size = System::Drawing::Size(859, 307);
this->groupBox3->TabIndex = 5;
this->groupBox3->TabStop = false;
this->groupBox3->Text = L"   ";
this->dataGridView1->AllowUserToAddRows = false;
this->dataGridView1->AllowUserToDeleteRows = false;
this->dataGridView1->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
this->dataGridView1->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
this->dataGridView1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->dataGridView1->Location = System::Drawing::Point(13, 21);
this->dataGridView1->Name = L"dataGridView1";
this->dataGridView1->ReadOnly = true;
this->dataGridView1->Size = System::Drawing::Size(840, 262);
this->dataGridView1->TabIndex = 0;
this->dataGridView1->CellContentClick += gcnew System::Windows::Forms::DataGridViewCellEventHandler(this, &MyForm::dataGridView1_CellContentClick);
this->groupBox1->BackColor = System::Drawing::Color::RosyBrown;
this->groupBox1->Controls->Add(this->label20);
this->groupBox1->Controls->Add(this->label19);
this->groupBox1->Controls->Add(this->label18);
this->groupBox1->Controls->Add(this->label17);
this->groupBox1->Controls->Add(this->label16);
this->groupBox1->Controls->Add(this->label10);
this->groupBox1->Controls->Add(this->label9);
this->groupBox1->Controls->Add(this->label8);
this->groupBox1->Controls->Add(this->label7);
this->groupBox1->Controls->Add(this->label6);
this->groupBox1->FlatStyle = System::Windows::Forms::FlatStyle::Popup;
this->groupBox1->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->groupBox1->Location = System::Drawing::Point(3, 336);
this->groupBox1->Name = L"groupBox1";
this->groupBox1->Size = System::Drawing::Size(397, 201);
this->groupBox1->TabIndex = 3;
this->groupBox1->TabStop = false;
this->groupBox1->Text = L"   ";
this->groupBox1->Enter += gcnew System::EventHandler(this, &MyForm::groupBox1_Enter);
this->label20->BackColor = System::Drawing::Color::Silver;
this->label20->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label20->Location = System::Drawing::Point(198, 170);
this->label20->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label20->Name = L"label20";
this->label20->Size = System::Drawing::Size(193, 24);
this->label20->TabIndex = 25;
this->label20->Text = L"     ";
this->label20->Visible = false;
this->label19->BackColor = System::Drawing::Color::Silver;
this->label19->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label19->Location = System::Drawing::Point(198, 125);
this->label19->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label19->Name = L"label19";
this->label19->Size = System::Drawing::Size(193, 24);
this->label19->TabIndex = 24;
this->label19->Text = L"     ";
this->label19->Visible = false;
this->label18->BackColor = System::Drawing::Color::Silver;
this->label18->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label18->Location = System::Drawing::Point(198, 93);
this->label18->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label18->Name = L"label18";
this->label18->Size = System::Drawing::Size(193, 24);
this->label18->TabIndex = 23;
this->label18->Text = L"     ";
this->label18->Visible = false;
this->label17->BackColor = System::Drawing::Color::Silver;
this->label17->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label17->Location = System::Drawing::Point(198, 58);
this->label17->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label17->Name = L"label17";
this->label17->Size = System::Drawing::Size(193, 24);
this->label17->TabIndex = 22;
this->label17->Text = L"     ";
this->label17->Visible = false;
this->label16->BackColor = System::Drawing::Color::Silver;
this->label16->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label16->Location = System::Drawing::Point(198, 25);
this->label16->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label16->Name = L"label16";
this->label16->Size = System::Drawing::Size(193, 24);
this->label16->TabIndex = 21;
this->label16->Text = L"     ";
this->label16->Visible = false;
this->label10->AutoSize = true;
this->label10->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label10->Location = System::Drawing::Point(6, 164);
this->label10->Name = L"label10";
this->label10->Size = System::Drawing::Size(186, 30);
this->label10->TabIndex = 5;
this->label10->Text = L" \r\n , ";
this->label9->AutoSize = true;
this->label9->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label9->Location = System::Drawing::Point(6, 119);
this->label9->Name = L"label9";
this->label9->Size = System::Drawing::Size(135, 30);
this->label9->TabIndex = 4;
this->label9->Text = L" \r\n( )\r\n";
this->label8->AutoSize = true;
this->label8->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label8->Location = System::Drawing::Point(6, 94);
this->label8->Name = L"label8";
this->label8->Size = System::Drawing::Size(145, 15);
this->label8->TabIndex = 3;
this->label8->Text = L" ";
this->label7->AutoSize = true;
this->label7->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label7->Location = System::Drawing::Point(6, 59);
this->label7->Name = L"label7";
this->label7->Size = System::Drawing::Size(157, 15);
this->label7->TabIndex = 2;
this->label7->Text = L" ";
this->label6->AutoSize = true;
this->label6->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label6->Location = System::Drawing::Point(8, 26);
this->label6->Name = L"label6";
this->label6->Size = System::Drawing::Size(117, 15);
this->label6->TabIndex = 1;
this->label6->Text = L" ";
this->panel2->BackColor = System::Drawing::Color::LightGray;
this->panel2->Controls->Add(this->groupBox2);
this->panel2->Location = System::Drawing::Point(3, 354);
this->panel2->Name = L"panel2";
this->panel2->Size = System::Drawing::Size(406, 376);
this->panel2->TabIndex = 2;
this->groupBox2->BackColor = System::Drawing::SystemColors::ActiveCaption;
this->groupBox2->Controls->Add(this->label25);
this->groupBox2->Controls->Add(this->label24);
this->groupBox2->Controls->Add(this->label23);
this->groupBox2->Controls->Add(this->label22);
this->groupBox2->Controls->Add(this->label21);
this->groupBox2->Controls->Add(this->label11);
this->groupBox2->Controls->Add(this->label12);
this->groupBox2->Controls->Add(this->label13);
this->groupBox2->Controls->Add(this->label14);
this->groupBox2->Controls->Add(this->label15);
this->groupBox2->FlatStyle = System::Windows::Forms::FlatStyle::Popup;
this->groupBox2->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->groupBox2->Location = System::Drawing::Point(3, 181);
this->groupBox2->Name = L"groupBox2";
this->groupBox2->Size = System::Drawing::Size(400, 193);
this->groupBox2->TabIndex = 4;
this->groupBox2->TabStop = false;
this->groupBox2->Text = L"   ";
this->label25->BackColor = System::Drawing::Color::AliceBlue;
this->label25->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label25->Location = System::Drawing::Point(195, 149);
this->label25->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label25->Name = L"label25";
this->label25->Size = System::Drawing::Size(193, 24);
this->label25->TabIndex = 26;
this->label25->Text = L"     ";
this->label25->Visible = false;
this->label24->BackColor = System::Drawing::Color::AliceBlue;
this->label24->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label24->Location = System::Drawing::Point(195, 113);
this->label24->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label24->Name = L"label24";
this->label24->Size = System::Drawing::Size(193, 24);
this->label24->TabIndex = 25;
this->label24->Text = L"     ";
this->label24->Visible = false;
this->label23->BackColor = System::Drawing::Color::AliceBlue;
this->label23->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label23->Location = System::Drawing::Point(195, 82);
this->label23->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label23->Name = L"label23";
this->label23->Size = System::Drawing::Size(193, 24);
this->label23->TabIndex = 24;
this->label23->Text = L"     ";
this->label23->Visible = false;
this->label22->BackColor = System::Drawing::Color::AliceBlue;
this->label22->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label22->Location = System::Drawing::Point(195, 54);
this->label22->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label22->Name = L"label22";
this->label22->Size = System::Drawing::Size(193, 24);
this->label22->TabIndex = 23;
this->label22->Text = L"     ";
this->label22->Visible = false;
this->label21->BackColor = System::Drawing::Color::AliceBlue;
this->label21->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->label21->Location = System::Drawing::Point(195, 26);
this->label21->Margin = System::Windows::Forms::Padding(5, 0, 5, 0);
this->label21->Name = L"label21";
this->label21->Size = System::Drawing::Size(193, 24);
this->label21->TabIndex = 22;
this->label21->Text = L"     ";
this->label21->Visible = false;
this->label11->AutoSize = true;
this->label11->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label11->Location = System::Drawing::Point(5, 143);
this->label11->Name = L"label11";
this->label11->Size = System::Drawing::Size(186, 30);
this->label11->TabIndex = 5;
this->label11->Text = L" \r\n , ";
this->label12->AutoSize = true;
this->label12->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label12->Location = System::Drawing::Point(7, 107);
this->label12->Name = L"label12";
this->label12->Size = System::Drawing::Size(135, 30);
this->label12->TabIndex = 4;
this->label12->Text = L" \r\n( )";
this->label13->AutoSize = true;
this->label13->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label13->Location = System::Drawing::Point(5, 83);
this->label13->Name = L"label13";
this->label13->Size = System::Drawing::Size(145, 15);
this->label13->TabIndex = 3;
this->label13->Text = L" ";
this->label14->AutoSize = true;
this->label14->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label14->Location = System::Drawing::Point(5, 55);
this->label14->Name = L"label14";
this->label14->Size = System::Drawing::Size(157, 15);
this->label14->TabIndex = 2;
this->label14->Text = L" ";
this->label15->AutoSize = true;
this->label15->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label15->Location = System::Drawing::Point(7, 27);
this->label15->Name = L"label15";
this->label15->Size = System::Drawing::Size(117, 15);
this->label15->TabIndex = 1;
this->label15->Text = L" ";
this->panel1->BackColor = System::Drawing::Color::DarkGray;
this->panel1->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
this->panel1->Controls->Add(this->button2);
this->panel1->Controls->Add(this->button1);
this->panel1->Controls->Add(this->numericUpDown5);
this->panel1->Controls->Add(this->label5);
this->panel1->Controls->Add(this->numericUpDown4);
this->panel1->Controls->Add(this->numericUpDown3);
this->panel1->Controls->Add(this->numericUpDown2);
this->panel1->Controls->Add(this->numericUpDown1);
this->panel1->Controls->Add(this->label4);
this->panel1->Controls->Add(this->label3);
this->panel1->Controls->Add(this->label2);
this->panel1->Controls->Add(this->label1);
this->panel1->Location = System::Drawing::Point(3, 6);
this->panel1->Name = L"panel1";
this->panel1->RightToLeft = System::Windows::Forms::RightToLeft::No;
this->panel1->Size = System::Drawing::Size(397, 332);
this->panel1->TabIndex = 1;
this->button2->BackColor = System::Drawing::SystemColors::ActiveCaption;
this->button2->FlatStyle = System::Windows::Forms::FlatStyle::Popup;
this->button2->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->button2->Location = System::Drawing::Point(9, 274);
this->button2->Name = L"button2";
this->button2->Size = System::Drawing::Size(225, 38);
this->button2->TabIndex = 11;
this->button2->Text = L" \r\n ";
this->button2->UseVisualStyleBackColor = false;
this->button2->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
this->button1->BackColor = System::Drawing::Color::RosyBrown;
this->button1->FlatStyle = System::Windows::Forms::FlatStyle::Popup;
this->button1->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->button1->Location = System::Drawing::Point(9, 216);
this->button1->Name = L"button1";
this->button1->Size = System::Drawing::Size(225, 38);
this->button1->TabIndex = 10;
this->button1->Text = L"  ";
this->button1->UseVisualStyleBackColor = false;
this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
this->numericUpDown5->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->numericUpDown5->Location = System::Drawing::Point(248, 172);
this->numericUpDown5->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8, 0, 0, 0 });
this->numericUpDown5->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
this->numericUpDown5->Name = L"numericUpDown5";
this->numericUpDown5->Size = System::Drawing::Size(102, 21);
this->numericUpDown5->TabIndex = 9;
this->numericUpDown5->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 2, 0, 0, 0 });
this->label5->AutoSize = true;
this->label5->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label5->Location = System::Drawing::Point(14, 162);
this->label5->Name = L"label5";
this->label5->Size = System::Drawing::Size(228, 30);
this->label5->TabIndex = 8;
this->label5->Text = L"  \r\n(  )\r\n";
this->numericUpDown4->DecimalPlaces = 8;
this->numericUpDown4->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->numericUpDown4->Location = System::Drawing::Point(201, 128);
this->numericUpDown4->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 327680 });
this->numericUpDown4->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 983040 });
this->numericUpDown4->Name = L"numericUpDown4";
this->numericUpDown4->Size = System::Drawing::Size(149, 21);
this->numericUpDown4->TabIndex = 7;
this->numericUpDown4->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 524288 });
this->numericUpDown3->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->numericUpDown3->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
this->numericUpDown3->Location = System::Drawing::Point(201, 86);
this->numericUpDown3->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 30000, 0, 0, 0 });
this->numericUpDown3->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 50, 0, 0, 0 });
this->numericUpDown3->Name = L"numericUpDown3";
this->numericUpDown3->Size = System::Drawing::Size(149, 21);
this->numericUpDown3->TabIndex = 6;
this->numericUpDown3->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
this->numericUpDown2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->numericUpDown2->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
this->numericUpDown2->Location = System::Drawing::Point(201, 47);
this->numericUpDown2->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
this->numericUpDown2->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
this->numericUpDown2->Name = L"numericUpDown2";
this->numericUpDown2->Size = System::Drawing::Size(149, 21);
this->numericUpDown2->TabIndex = 5;
this->numericUpDown2->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
this->numericUpDown1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9));
this->numericUpDown1->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
this->numericUpDown1->Location = System::Drawing::Point(201, 16);
this->numericUpDown1->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
this->numericUpDown1->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
this->numericUpDown1->Name = L"numericUpDown1";
this->numericUpDown1->Size = System::Drawing::Size(149, 21);
this->numericUpDown1->TabIndex = 4;
this->numericUpDown1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
this->label4->AutoSize = true;
this->label4->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label4->Location = System::Drawing::Point(14, 119);
this->label4->Name = L"label4";
this->label4->Size = System::Drawing::Size(166, 30);
this->label4->TabIndex = 3;
this->label4->Text = L" \r\n  , eps";
this->label3->AutoSize = true;
this->label3->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label3->Location = System::Drawing::Point(14, 77);
this->label3->Name = L"label3";
this->label3->Size = System::Drawing::Size(173, 30);
this->label3->TabIndex = 2;
this->label3->Text = L"  \r\n  , Nmax";
this->label2->AutoSize = true;
this->label2->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label2->Location = System::Drawing::Point(14, 50);
this->label2->Name = L"label2";
this->label2->Size = System::Drawing::Size(168, 15);
this->label2->TabIndex = 1;
this->label2->Text = L"   y, m";
this->label1->AutoSize = true;
this->label1->Font = (gcnew System::Drawing::Font(L"Georgia", 9, System::Drawing::FontStyle::Bold));
this->label1->Location = System::Drawing::Point(14, 19);
this->label1->Name = L"label1";
this->label1->Size = System::Drawing::Size(164, 15);
this->label1->TabIndex = 0;
this->label1->Text = L"   x, n";
this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.Image")));
this->pictureBox1->Location = System::Drawing::Point(402, 6);
this->pictureBox1->Name = L"pictureBox1";
this->pictureBox1->Size = System::Drawing::Size(264, 132);
this->pictureBox1->TabIndex = 0;
this->pictureBox1->TabStop = false;
this->tabPage2->Location = System::Drawing::Point(4, 22);
this->tabPage2->Name = L"tabPage2";
this->tabPage2->Padding = System::Windows::Forms::Padding(3);
this->tabPage2->Size = System::Drawing::Size(1269, 731);
this->tabPage2->TabIndex = 1;
this->tabPage2->Text = L" ";
this->tabPage2->UseVisualStyleBackColor = true;
this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
this->ClientSize = System::Drawing::Size(1266, 755);
this->Controls->Add(this->tabControl1);
this->Name = L"MyForm";
this->Text = L"     ,    "
L"";
this->tabControl1->ResumeLayout(false);
this->tabPage1->ResumeLayout(false);
this->groupBox4->ResumeLayout(false);
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->dataGridView2))->EndInit();
this->groupBox3->ResumeLayout(false);
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->dataGridView1))->EndInit();
this->groupBox1->ResumeLayout(false);
this->groupBox1->PerformLayout();
this->panel2->ResumeLayout(false);
this->groupBox2->ResumeLayout(false);
this->groupBox2->PerformLayout();
this->panel1->ResumeLayout(false);
this->panel1->PerformLayout();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown5))->EndInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->EndInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->EndInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->EndInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->EndInit();
(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
this->ResumeLayout(false);

}
#pragma endregion
private: System::Void tabPage1_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void groupBox1_Enter(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void dataGridView1_CellContentClick(System::Object^ sender, System::Windows::Forms::DataGridViewCellEventArgs^ e) {
}
private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {


size_t n = Convert::ToInt32(numericUpDown1->Text);
size_t m = Convert::ToInt32(numericUpDown2->Text);

int a = -1, b = 1, c = -1, d = 1;


double h = (b - a) / (double)n;
double k = (d - c) / (double)m;

double param_x = -h * h;
double param_y = -k * k;
double A = -2 * (1 / param_x + 1 / param_y);

matrix V, r, H;
V.assign(m + 1, vector<double>(n + 1));
r.assign(m, vector<double>(n));
H.assign(m, vector<double>(n));

for (int i = 0; i < m + 1; i++) {
V[i][0] = mu1(c + i * k);
}

for (int i = 0; i < m + 1; i++) {
V[i][n] = mu2(c + i * k);
}

for (int i = 0; i < n + 1; i++) {
V[0][i] = mu3(a + i * h);
}

for (int i = 0; i < n + 1; i++) {
V[m][i] = mu4(a + i * h);
}


double alpha, Ahh = 1;
int step_count = 0;
int max_step = Convert::ToInt32(numericUpDown3->Text);;
double eps = Convert::ToDouble(numericUpDown4->Text);;
double eps_curr = 0;
double accuracy = 0;
double v_old;
double runtime = clock();
do
{
accuracy = 0;
r = calcDiscreapancy(r, V, param_x, param_y, A, a, b, c, d, n, m);

alpha = calcAlpha(H, r, param_x, param_y, A, Ahh, n, m);

for (int i = 1; i < m; i++) {
for (int j = 1; j < n; j++) {
v_old = V[i][j];
V[i][j] = v_old + alpha * H[i - 1][j - 1];
eps_curr = abs(V[i][j] - v_old);
if (eps_curr > accuracy) {
accuracy = eps_curr;
}
}
}

step_count++;
} while ((accuracy > eps) && (step_count < max_step));
runtime = (clock() - runtime) / CLOCKS_PER_SEC;

r = calcDiscreapancy(r, V, param_x, param_y, A, a, b, c, d, n, m);

double Disc_max = 0;
for (int i = 0; i < r.size(); i++) {
for (int j = 0; j < r[0].size(); j++) {
if (r[i][j] > Disc_max) {
Disc_max = r[i][j];
}
}
}

double error = calcError(V, h, k, a, c);


label16->Visible = true;
label17->Visible = true;
label18->Visible = true;
label19->Visible = true;
label20->Visible = true;

label16->Text = Convert::ToString(step_count);
label17->Text = Convert::ToString(accuracy);
label18->Text = Convert::ToString(error);
label19->Text = Convert::ToString(Disc_max);
label20->Text = Convert::ToString(runtime);



dataGridView1->Rows->Clear();

dataGridView1->RowCount = m + 2;
dataGridView1->ColumnCount = n + 3;
dataGridView1->RowHeadersVisible = false;
for (int i = n, col = 2; i >= 0; i--, col++) {
dataGridView1->Columns[col]->HeaderText = Convert::ToString(n - i);
dataGridView1->Rows[0]->Cells[i + 2]->Value = ceil((a + i * h) * 1000) / 1000;
}

for (int i = m, row = 1; i >= 0; i--, row++) {
dataGridView1->Rows[row]->Cells[0]->Value = i;
dataGridView1->Rows[row]->Cells[1]->Value = ceil((c + i * k) * 1000) / 1000;
}

dataGridView1->Rows[0]->Cells[0]->Value = Convert::ToString("j");
dataGridView1->Columns[1]->HeaderText = Convert::ToString("i");
dataGridView1->Rows[0]->Cells[1]->Value = Convert::ToString("Y / X");

for (int i = m + 1, k = 0; i > 0; i--, k++) {
for (int j = 2, p = 0; j < n + 3; j++, p++) {
dataGridView1->Rows[i]->Cells[j]->Value = ceil(V[k][p] * 1000) / 1000;
}
}
}


















private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {

size_t n = Convert::ToInt32(numericUpDown1->Text);
size_t m = Convert::ToInt32(numericUpDown2->Text);

int a = -1, b = 1, c = -1, d = 1;


double h = (b - a) / (double)n;
double k = (d - c) / (double)m;

double param_x = -h * h;
double param_y = -k * k;
double A = -2 * (1 / param_x + 1 / param_y);

matrix V, r, H;
V.assign(m + 1, vector<double>(n + 1));
r.assign(m, vector<double>(n));
H.assign(m, vector<double>(n));

for (int i = 0; i < m + 1; i++) {
V[i][0] = mu1(c + i * k);
}

for (int i = 0; i < m + 1; i++) {
V[i][n] = mu2(c + i * k);
}

for (int i = 0; i < n + 1; i++) {
V[0][i] = mu3(a + i * h);
}

for (int i = 0; i < n + 1; i++) {
V[m][i] = mu4(a + i * h);
}


double alpha, Ahh = 1;
int step_count = 0;
int max_step = Convert::ToInt32(numericUpDown3->Text);
double eps = Convert::ToDouble(numericUpDown4->Text);
double eps_curr = 0;
double accuracy = 0;
double v_old;
size_t num_threads = Convert::ToInt32(numericUpDown5->Text);
double parallelruntime = omp_get_wtime();
do
{
accuracy = 0;
r = parallelCalcDiscrepancy(r, V, param_x, param_y, A, a, b, c, d, n, m, num_threads);

alpha = parallelCalcAlpha(H, r, param_x, param_y, A, Ahh, n, m, num_threads);

for (int i = 1; i < m; i++) {
for (int j = 1; j < n; j++) {
v_old = V[i][j];
V[i][j] = v_old + alpha * H[i - 1][j - 1];
eps_curr = abs(V[i][j] - v_old);
if (eps_curr > accuracy) {
accuracy = eps_curr;
}
}
}

step_count++;
} while ((accuracy > eps) && (step_count < max_step));
parallelruntime = omp_get_wtime() - parallelruntime;

r = parallelCalcDiscrepancy(r, V, param_x, param_y, A, a, b, c, d, n, m, num_threads);

double Disc_max = 0;
for (int i = 0; i < r.size(); i++) {
for (int j = 0; j < r[0].size(); j++) {
if (r[i][j] > Disc_max) {
Disc_max = r[i][j];
}
}
}

double error = calcError(V, h, k, a, c);


label21->Visible = true;
label22->Visible = true;
label23->Visible = true;
label24->Visible = true;
label25->Visible = true;

label21->Text = Convert::ToString(step_count);
label22->Text = Convert::ToString(accuracy);
label23->Text = Convert::ToString(error);
label24->Text = Convert::ToString(Disc_max);
label25->Text = Convert::ToString(parallelruntime);

dataGridView2->Rows->Clear();
dataGridView2->RowCount = m + 2;
dataGridView2->ColumnCount = n + 3;
dataGridView2->RowHeadersVisible = false;
for (int i = n, col = 2; i >= 0; i--, col++) {
dataGridView2->Columns[col]->HeaderText = Convert::ToString(n - i);
dataGridView2->Rows[0]->Cells[i + 2]->Value = ceil((a + i * h) * 1000) / 1000;
}

for (int i = m, row = 1; i >= 0; i--, row++) {
dataGridView2->Rows[row]->Cells[0]->Value = i;
dataGridView2->Rows[row]->Cells[1]->Value = ceil((c + i * k) * 1000) / 1000;
}

dataGridView2->Rows[0]->Cells[0]->Value = Convert::ToString("j");
dataGridView2->Columns[1]->HeaderText = Convert::ToString("i");
dataGridView2->Rows[0]->Cells[1]->Value = Convert::ToString("Y / X");

for (int i = m + 1, k = 0; i > 0; i--, k++) {
for (int j = 2, p = 0; j < n + 3; j++, p++) {
dataGridView2->Rows[i]->Cells[j]->Value = ceil(V[k][p] * 1000) / 1000;
}
}
}
};

};
