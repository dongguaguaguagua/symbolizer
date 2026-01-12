import 'package:flutter/material.dart';
import 'package:mobile/pages/infer_page.dart';
import 'package:mobile/pages/submit_page.dart';
import 'package:mobile/pages/symbol_list_page.dart';
import 'package:flutter_svg/flutter_svg.dart';

void main() {
	svg.cache.maximumSize = 2000;
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Symbol Collector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const AppShell(),
    );
  }
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;

  final _pages = const [
    InferPage(),
    SubmitPage(),
    SymbolListPage(),
  ];


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(child: _pages[_index]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.auto_fix_high),
            label: '识别',
          ),
          NavigationDestination(
            icon: Icon(Icons.upload),
            label: '提交',
          ),
          NavigationDestination(
            icon: Icon(Icons.list),
            label: '符号',
          ),
        ],
      ),
    );
  }
}
